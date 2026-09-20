from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
MART_PATH = ROOT / "data" / "processed" / "order_fulfillment_mart.parquet"
OUTPUT_DIR = ROOT / "outputs" / "model"
MODEL_DIR = ROOT / "models"

PUSH_FEATURES = [
    "da_id",
    "push_hour",
    "push_weekday",
    "is_weekend",
    "merchant_customer_distance_km",
    "promise_minutes",
    "order_to_push_minutes",
]

ACCEPTANCE_FEATURES = PUSH_FEATURES + [
    "waybill_attempt_count",
    "rejected_waybill_count",
    "first_dispatch_wait_minutes",
    "accepted_dispatch_wait_minutes",
    "accepted_dispatch_to_grab_minutes",
    "courier_merchant_distance_at_accept_km",
]

CATEGORICAL_FEATURES = ["da_id", "push_hour", "push_weekday"]


def prepare_features(
    data: pd.DataFrame, features: list[str], categories: dict[str, list] | None = None
) -> tuple[pd.DataFrame, dict[str, list]]:
    result = data[features].copy()
    if categories is None:
        categories = {
            column: sorted(result[column].dropna().unique().tolist())
            for column in CATEGORICAL_FEATURES
            if column in features
        }
    for column, levels in categories.items():
        if column in features:
            known = result[column].where(result[column].isin(levels))
            result[column] = pd.Categorical(known, categories=levels)
    return result, categories


def build_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.08,
        max_iter=180,
        max_leaf_nodes=31,
        min_samples_leaf=100,
        l2_regularization=1.0,
        categorical_features="from_dtype",
        early_stopping=False,
        random_state=42,
    )


def ranking_metrics(
    y_true: pd.Series,
    probability: np.ndarray,
    top_fraction: float = 0.10,
    calibrated_probability: bool = True,
) -> dict:
    y = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    k = max(1, math.ceil(len(y) * top_fraction))
    top_index = np.argsort(-probability, kind="stable")[:k]
    total_positive = y.sum()
    top_positive = y[top_index].sum()
    prevalence = y.mean()
    top_precision = top_positive / k
    return {
        "rows": int(len(y)),
        "late_orders": int(total_positive),
        "late_rate": float(prevalence),
        "roc_auc": float(roc_auc_score(y, probability)),
        "pr_auc": float(average_precision_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)) if calibrated_probability else np.nan,
        "recall_at_top_10pct": float(top_positive / total_positive),
        "precision_at_top_10pct": float(top_precision),
        "lift_at_top_10pct": float(top_precision / prevalence),
    }


def fit_and_score(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    features: list[str],
) -> tuple[HistGradientBoostingClassifier, np.ndarray, dict[str, list]]:
    x_train, categories = prepare_features(train, features)
    x_eval, _ = prepare_features(evaluation, features, categories)
    y_train = train["is_late"].astype(int)
    model = build_model()
    model.fit(x_train, y_train)
    probability = model.predict_proba(x_eval)[:, 1]
    return model, probability, categories


def main() -> None:
    orders = pd.read_parquet(MART_PATH)
    data = orders.loc[orders["is_strategy_eligible"]].copy()
    data["is_late"] = data["is_late"].astype(bool)

    candidate_train = data.loc[data["dt"].le(20221021)].copy()
    validation = data.loc[data["dt"].eq(20221022)].copy()
    final_train = data.loc[data["dt"].le(20221022)].copy()
    test = data.loc[data["dt"].ge(20221023)].copy()

    assert candidate_train["dt"].max() < validation["dt"].min()
    assert final_train["dt"].max() < test["dt"].min()
    assert len(candidate_train) + len(validation) + len(test) == len(data)

    metric_rows = []
    validation_predictions = {}
    for model_name, features in [
        ("push_time_model", PUSH_FEATURES),
        ("acceptance_time_model", ACCEPTANCE_FEATURES),
    ]:
        _, probability, _ = fit_and_score(candidate_train, validation, features)
        validation_predictions[model_name] = probability
        metrics = ranking_metrics(validation["is_late"], probability)
        metric_rows.append({"split": "validation_20221022", "model": model_name, **metrics})

    final_models = {}
    final_categories = {}
    test_predictions = test[
        ["order_id", "dt", "da_id", "push_time_slot_30m", "is_late"]
    ].copy()
    for model_name, features in [
        ("push_time_model", PUSH_FEATURES),
        ("acceptance_time_model", ACCEPTANCE_FEATURES),
    ]:
        model, probability, categories = fit_and_score(final_train, test, features)
        final_models[model_name] = model
        final_categories[model_name] = categories
        test_predictions[f"{model_name}_risk"] = probability
        metrics = ranking_metrics(test["is_late"], probability)
        metric_rows.append({"split": "test_20221023_20221024", "model": model_name, **metrics})

    # Distance-only benchmark: useful to prove the model adds signal beyond a
    # single obvious feature without pretending this is a calibrated model.
    distance_score = test["merchant_customer_distance_km"].to_numpy()
    distance_metrics = ranking_metrics(
        test["is_late"], distance_score, calibrated_probability=False
    )
    metric_rows.append(
        {
            "split": "test_20221023_20221024",
            "model": "distance_only_ranking",
            **distance_metrics,
        }
    )

    metrics = pd.DataFrame(metric_rows)

    acceptance_model = final_models["acceptance_time_model"]
    acceptance_categories = final_categories["acceptance_time_model"]
    importance_sample = test.sample(n=min(50_000, len(test)), random_state=42)
    x_importance, _ = prepare_features(
        importance_sample, ACCEPTANCE_FEATURES, acceptance_categories
    )
    importance = permutation_importance(
        acceptance_model,
        x_importance,
        importance_sample["is_late"].astype(int),
        scoring="average_precision",
        n_repeats=3,
        random_state=42,
        # Keep the pipeline portable on memory-constrained laptops. Spawning one
        # worker per CPU can duplicate the 50k-row evaluation frame on Windows.
        n_jobs=1,
    )
    importance_table = pd.DataFrame(
        {
            "feature": ACCEPTANCE_FEATURES,
            "pr_auc_importance_mean": importance.importances_mean,
            "pr_auc_importance_std": importance.importances_std,
        }
    ).sort_values("pr_auc_importance_mean", ascending=False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False, encoding="utf-8-sig")
    test_predictions.to_parquet(
        OUTPUT_DIR / "test_predictions.parquet", index=False, compression="zstd"
    )
    importance_table.to_csv(
        OUTPUT_DIR / "acceptance_model_permutation_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    joblib.dump(
        {
            "model": final_models["push_time_model"],
            "features": PUSH_FEATURES,
            "categories": final_categories["push_time_model"],
            "prediction_time": "order_push_time",
        },
        MODEL_DIR / "push_time_late_risk.joblib",
    )
    joblib.dump(
        {
            "model": final_models["acceptance_time_model"],
            "features": ACCEPTANCE_FEATURES,
            "categories": final_categories["acceptance_time_model"],
            "prediction_time": "grab_time",
        },
        MODEL_DIR / "acceptance_time_late_risk.joblib",
    )

    profile = {
        "date_splits": {
            "candidate_train": [int(candidate_train["dt"].min()), int(candidate_train["dt"].max())],
            "validation": [int(validation["dt"].min()), int(validation["dt"].max())],
            "final_train": [int(final_train["dt"].min()), int(final_train["dt"].max())],
            "test": [int(test["dt"].min()), int(test["dt"].max())],
        },
        "row_counts": {
            "candidate_train": int(len(candidate_train)),
            "validation": int(len(validation)),
            "final_train": int(len(final_train)),
            "test": int(len(test)),
        },
        "metrics": json.loads(metrics.to_json(orient="records")),
        "top_features": json.loads(importance_table.head(10).to_json(orient="records")),
    }
    (OUTPUT_DIR / "model_profile.json").write_text(
        json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(metrics.to_string(index=False))
    print("\nTop permutation importance:\n", importance_table.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
