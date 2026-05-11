import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config_loader import load_config
from app.core.registry import load_model_registry, get_active_model_info
from app.model.save_load import load_model
from app.model.predict import predict_proba


def load_real_transactions(project_root: Path) -> pd.DataFrame:
    dataset_path = project_root / "data" / "raw" / "AIML Dataset.csv"

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)

    required_columns = [
        "step",
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud",
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[required_columns].copy()
    df = df.sort_values("step").reset_index(drop=True)

    df["transaction_id"] = df.index

    # Synthetic entity groups for suppression.
    df["entity_id"] = (
        df["type"].astype(str)
        + "_"
        + df["amount"].round(-2).astype(int).astype(str)
    )

    return df


def add_scores(df: pd.DataFrame, model) -> pd.DataFrame:
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
    df["expected_fraud_loss"] = df["fraud_score"] * df["amount"]
    df["expected_investigation_cost"] = (1 - df["fraud_score"]) * 50
    df["rank_score"] = (
        df["expected_fraud_loss"] - df["expected_investigation_cost"]
    )

    return df


def run_sequential_simulation(
    df: pd.DataFrame,
    alert_budget_per_step: int,
    suppression_window: int,
) -> pd.DataFrame:
    results = []
    recent_entities = {}

    for step, step_df in df.groupby("step", sort=True):
        step_df = step_df.copy()
        step_df["decision"] = 0
        step_df["suppressed"] = 0
        step_df["strategy"] = "sequential_full_system"

        # Remove expired entities from suppression memory
        expired_entities = [
            entity for entity, last_seen_step in recent_entities.items()
            if step - last_seen_step > suppression_window
        ]

        for entity in expired_entities:
            del recent_entities[entity]

        # Rank current step transactions
        step_df = step_df.sort_values("rank_score", ascending=False)

        alerts_used = 0

        for idx, row in step_df.iterrows():
            if alerts_used >= alert_budget_per_step:
                break

            entity_id = row["entity_id"]

            if entity_id in recent_entities:
                step_df.at[idx, "suppressed"] = 1
                continue

            step_df.at[idx, "decision"] = 1
            recent_entities[entity_id] = step
            alerts_used += 1

        results.append(step_df)

    return pd.concat(results).sort_index()


def evaluate_sequential_results(df: pd.DataFrame) -> dict:
    alerts = df[df["decision"] == 1]

    frauds_total = int(df["isFraud"].sum())
    alerts_count = int(len(alerts))
    frauds_caught = int(alerts["isFraud"].sum())
    missed_frauds = int(frauds_total - frauds_caught)
    false_positives = int(alerts_count - frauds_caught)

    precision = frauds_caught / alerts_count if alerts_count else 0
    recall = frauds_caught / frauds_total if frauds_total else 0
    alert_rate = alerts_count / len(df) if len(df) else 0

    missed_fraud_loss = df[
        (df["isFraud"] == 1) & (df["decision"] == 0)
    ]["amount"].sum()

    investigation_cost = alerts_count * 50
    total_cost = missed_fraud_loss + investigation_cost

    suppressed_count = int(df["suppressed"].sum())
    suppression_rate = suppressed_count / len(df) if len(df) else 0

    return {
        "strategy": "sequential_full_system",
        "transactions": len(df),
        "frauds_total": frauds_total,
        "alerts": alerts_count,
        "frauds_caught": frauds_caught,
        "missed_frauds": missed_frauds,
        "false_positives": false_positives,
        "precision_alerts": precision,
        "recall_fraud": recall,
        "alert_rate": alert_rate,
        "suppressed": suppressed_count,
        "suppression_rate": suppression_rate,
        "missed_fraud_loss": missed_fraud_loss,
        "investigation_cost": investigation_cost,
        "total_operational_cost": total_cost,
    }


def build_monitoring_windows(df: pd.DataFrame, window_size: int = 1000) -> pd.DataFrame:
    monitoring_rows = []

    for start in range(0, len(df), window_size):
        window = df.iloc[start:start + window_size]
        metrics = evaluate_sequential_results(window)
        metrics["window_start"] = start
        metrics["window_end"] = start + len(window) - 1
        monitoring_rows.append(metrics)

    return pd.DataFrame(monitoring_rows)


def main():
    print(">>> SEQUENTIAL SIMULATION RUNNING <<<")

    config = load_config()

    registry = load_model_registry(config["model"]["registry_path"])
    model_info = get_active_model_info(registry)
    model = load_model(model_info["path"])

    df = load_real_transactions(PROJECT_ROOT).head(10000)
    df = add_scores(df, model)

    alert_budget_per_step = 30
    suppression_window = 10

    print(f"\nTransactions evaluated: {len(df)}")
    print(f"Unique steps: {df['step'].nunique()}")
    print(f"Unique entities: {df['entity_id'].nunique()}")
    print(f"Alert budget per step: {alert_budget_per_step}")
    print(f"Suppression window: {suppression_window} steps")

    simulated_df = run_sequential_simulation(
        df=df,
        alert_budget_per_step=alert_budget_per_step,
        suppression_window=suppression_window,
    )

    summary = evaluate_sequential_results(simulated_df)
    summary_df = pd.DataFrame([summary])

    print("\nSEQUENTIAL SIMULATION SUMMARY")
    print(summary_df.to_string(index=False))

    monitoring_df = build_monitoring_windows(simulated_df, window_size=1000)

    print("\nMONITORING WINDOWS")
    print(monitoring_df[
        [
            "window_start",
            "window_end",
            "alerts",
            "frauds_caught",
            "precision_alerts",
            "recall_fraud",
            "suppressed",
            "total_operational_cost",
        ]
    ].to_string(index=False))

    output_summary_path = PROJECT_ROOT / "logs" / "sequential_simulation_summary.csv"
    output_monitoring_path = PROJECT_ROOT / "logs" / "sequential_monitoring_windows.csv"

    output_summary_path.parent.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(output_summary_path, index=False)
    monitoring_df.to_csv(output_monitoring_path, index=False)

    print(f"\nSaved summary to: {output_summary_path}")
    print(f"Saved monitoring windows to: {output_monitoring_path}")


if __name__ == "__main__":
    main()