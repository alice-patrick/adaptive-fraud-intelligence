import sys
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config_loader import load_config
from app.core.registry import load_model_registry, get_active_model_info
from app.model.save_load import load_model
from app.model.predict import predict_proba


app = FastAPI(
    title="Fraud Decision System API",
    docs_url="/docs",
    redoc_url="/redoc",
)


def load_data(limit: int = 1000) -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "raw" / "AIML Dataset.csv"

    df = pd.read_csv(path).head(limit)

    df = df[
        [
            "step",
            "type",
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
            "isFraud",
        ]
    ].copy()

    df = df.sort_values("step").reset_index(drop=True)

    df["transaction_id"] = df.index
    df["entity_id"] = (
        df["type"].astype(str)
        + "_"
        + df["amount"].round(-2).astype(int).astype(str)
    )

    return df


def load_active_model():
    config = load_config()
    registry = load_model_registry(config["model"]["registry_path"])
    model_info = get_active_model_info(registry)
    model = load_model(model_info["path"])
    return config, model


def score_data(df: pd.DataFrame, model) -> pd.DataFrame:
    feature_cols = [
        "step",
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
    ]

    scores = []

    for _, row in df.iterrows():
        transaction = row[feature_cols].to_dict()
        scores.append(predict_proba(model, transaction))

    df = df.copy()
    df["fraud_score"] = scores

    return df


def evaluate(df: pd.DataFrame, alert_col: str, investigation_cost: float) -> dict:
    alerts = df[df[alert_col] == 1]

    frauds_total = int(df["isFraud"].sum())
    alerts_count = int(len(alerts))
    frauds_caught = int(alerts["isFraud"].sum())

    missed_fraud_loss = df[
        (df["isFraud"] == 1) & (df[alert_col] == 0)
    ]["amount"].sum()

    total_cost = missed_fraud_loss + alerts_count * investigation_cost

    return {
        "alerts": alerts_count,
        "frauds_caught": frauds_caught,
        "recall": frauds_caught / frauds_total if frauds_total else 0,
        "precision": frauds_caught / alerts_count if alerts_count else 0,
        "total_cost": float(total_cost),
    }


def apply_ranking(
    df: pd.DataFrame,
    policy: str,
    investigation_cost: float,
) -> pd.DataFrame:
    df = df.copy()

    df["expected_benefit"] = (
        df["fraud_score"] * df["amount"] - investigation_cost
    )

    if policy == "score":
        df["rank_score"] = df["fraud_score"]

    elif policy == "benefit":
        df["rank_score"] = df["expected_benefit"]

    elif policy == "hybrid":
        df["rank_score"] = df["fraud_score"] * (df["amount"] ** 0.3)

    elif policy == "risk_zone":
        df["rank_score"] = df["fraud_score"] * (df["amount"] ** 0.3)

    else:
        raise ValueError(
            "Unknown policy. Use one of: score, benefit, hybrid, risk_zone."
        )

    return df


def run_decision_system(
    df: pd.DataFrame,
    investigation_cost: float,
    alert_budget: int,
    policy: str,
    static_threshold: float,
    risk_zone_floor: float = 0.3,
) -> pd.DataFrame:
    df = apply_ranking(df, policy, investigation_cost)

    df["alert"] = 0

    if policy == "risk_zone":
        high_risk = df[df["fraud_score"] >= static_threshold].copy()
        high_risk_indices = high_risk.sort_values(
            "fraud_score",
            ascending=False,
        ).index.tolist()

        selected_indices = []

        for idx in high_risk_indices:
            if len(selected_indices) >= alert_budget:
                break
            selected_indices.append(idx)

        remaining_budget = alert_budget - len(selected_indices)

        if remaining_budget > 0:
            candidate_df = df[
                (df["fraud_score"] >= risk_zone_floor)
                & (df["fraud_score"] < static_threshold)
                & (~df.index.isin(selected_indices))
            ].copy()

            candidate_indices = candidate_df.sort_values(
                "rank_score",
                ascending=False,
            ).head(remaining_budget).index.tolist()

            selected_indices.extend(candidate_indices)

        df.loc[selected_indices, "alert"] = 1
        return df

    ranked = df.sort_values("rank_score", ascending=False)
    selected_indices = ranked.head(alert_budget).index

    df.loc[selected_indices, "alert"] = 1

    return df


def run_static(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    df = df.copy()
    df["alert"] = (df["fraud_score"] >= threshold).astype(int)
    return df


def assign_severity(score: float) -> str:
    if score >= 0.95:
        return "HIGH"
    elif score >= 0.80:
        return "MEDIUM"
    return "LOW"


def build_reason(row) -> str:
    if row["severity"] == "HIGH":
        return "Very high model confidence; prioritized for immediate analyst review"
    elif row["severity"] == "MEDIUM":
        return "High model confidence; selected by decision policy and operational ranking"
    return "Risk-zone alert; lower model confidence but selected due to operational priority"


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/comparison")
def comparison(
    limit: int = 1000,
    investigation_cost: float = 10,
    ranking_policy: str = "score",
    risk_zone_floor: float = 0.3,
    budget_multiplier: float = 1.2,
):
    config, model = load_active_model()

    df = load_data(limit)
    df = score_data(df, model)

    threshold = config["decisioning"]["static_threshold"]

    static_df = run_static(df, threshold)
    static_alerts = int(static_df["alert"].sum())

    decision_budget = int(static_alerts * budget_multiplier)

    decision_df = run_decision_system(
        df=df,
        investigation_cost=investigation_cost,
        alert_budget=decision_budget,
        policy=ranking_policy,
        static_threshold=threshold,
        risk_zone_floor=risk_zone_floor,
    )

    static_metrics = evaluate(static_df, "alert", investigation_cost)
    decision_metrics = evaluate(decision_df, "alert", investigation_cost)

    return {
        "parameters": {
            "limit": limit,
            "investigation_cost": investigation_cost,
            "ranking_policy": ranking_policy,
            "risk_zone_floor": risk_zone_floor,
            "static_threshold": threshold,
            "static_alert_budget": static_alerts,
            "decision_budget": decision_budget,
            "budget_multiplier": budget_multiplier,
        },
        "static": static_metrics,
        "decision_system": decision_metrics,
        "cost_diff": decision_metrics["total_cost"] - static_metrics["total_cost"],
        "recall_diff": decision_metrics["recall"] - static_metrics["recall"],
        "precision_diff": decision_metrics["precision"] - static_metrics["precision"],
    }


@app.get("/alerts")
def alerts(
    limit: int = 3000,
    investigation_cost: float = 10,
    ranking_policy: str = "risk_zone",
    risk_zone_floor: float = 0.3,
    budget_multiplier: float = 1.4,
):
    config, model = load_active_model()

    df = load_data(limit)
    df = score_data(df, model)

    threshold = config["decisioning"]["static_threshold"]

    static_df = run_static(df, threshold)
    static_alerts = int(static_df["alert"].sum())
    decision_budget = int(static_alerts * budget_multiplier)

    decision_df = run_decision_system(
        df=df,
        investigation_cost=investigation_cost,
        alert_budget=decision_budget,
        policy=ranking_policy,
        static_threshold=threshold,
        risk_zone_floor=risk_zone_floor,
    )

    decision_df = apply_ranking(
        decision_df,
        policy=ranking_policy,
        investigation_cost=investigation_cost,
    )

    flagged = decision_df[decision_df["alert"] == 1].copy()

    flagged["severity"] = flagged["fraud_score"].apply(assign_severity)

    flagged["analyst_priority"] = flagged["severity"].map(
        {
            "HIGH": 1,
            "MEDIUM": 2,
            "LOW": 3,
        }
    )

    flagged["reason"] = flagged.apply(build_reason, axis=1)

    flagged = flagged.sort_values(
        ["analyst_priority", "rank_score"],
        ascending=[True, False],
    )

    return flagged[
        [
            "transaction_id",
            "step",
            "type",
            "amount",
            "fraud_score",
            "rank_score",
            "expected_benefit",
            "severity",
            "reason",
            "isFraud",
        ]
    ].head(200).to_dict(orient="records")


@app.get("/operating_curve")
def operating_curve(
    limit: int = 10000,
    investigation_cost: float = 10,
    ranking_policy: str = "risk_zone",
    risk_zone_floor: float = 0.3,
):
    config, model = load_active_model()

    df = load_data(limit)
    df = score_data(df, model)

    threshold = config["decisioning"]["static_threshold"]

    static_df = run_static(df, threshold)
    static_alerts = int(static_df["alert"].sum())
    static_metrics = evaluate(static_df, "alert", investigation_cost)

    rows = []

    for budget_multiplier in [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
        decision_budget = int(static_alerts * budget_multiplier)

        decision_df = run_decision_system(
            df=df,
            investigation_cost=investigation_cost,
            alert_budget=decision_budget,
            policy=ranking_policy,
            static_threshold=threshold,
            risk_zone_floor=risk_zone_floor,
        )

        decision_metrics = evaluate(
            decision_df,
            "alert",
            investigation_cost,
        )

        rows.append(
            {
                "budget_multiplier": budget_multiplier,
                "alert_budget": decision_budget,
                "alerts": decision_metrics["alerts"],
                "frauds_caught": decision_metrics["frauds_caught"],
                "recall": decision_metrics["recall"],
                "precision": decision_metrics["precision"],
                "total_cost": decision_metrics["total_cost"],
                "cost_diff_vs_static": (
                    decision_metrics["total_cost"]
                    - static_metrics["total_cost"]
                ),
                "recall_diff_vs_static": (
                    decision_metrics["recall"]
                    - static_metrics["recall"]
                ),
                "precision_diff_vs_static": (
                    decision_metrics["precision"]
                    - static_metrics["precision"]
                ),
            }
        )

    return {
        "parameters": {
            "limit": limit,
            "investigation_cost": investigation_cost,
            "ranking_policy": ranking_policy,
            "risk_zone_floor": risk_zone_floor,
            "static_threshold": threshold,
            "static_alerts": static_alerts,
        },
        "static_baseline": static_metrics,
        "operating_curve": rows,
    }