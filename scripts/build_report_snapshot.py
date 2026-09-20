from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
SNAPSHOT_PATH = OUT / "report_snapshot.json"
OFFICIAL_URL = "https://github.com/meituan/Meituan-INFORMS-TSL-Research-Challenge"


def records(frame: pd.DataFrame) -> list[dict]:
    clean = frame.astype(object).where(pd.notna(frame), None)
    return clean.to_dict(orient="records")


def source(label: str, files: list[str], filters: list[str], definitions: list[dict], methods=None):
    return {
        "label": label,
        "url": OFFICIAL_URL,
        "files": files,
        "filters": filters,
        "executedAt": datetime.now(timezone.utc).isoformat(),
        "metricDefinitions": definitions,
        "evidenceFlow": [
            {
                "title": "Official public source",
                "detail": "Meituan-INFORMS-TSL Research Challenge public dataset and official dataset paper.",
            },
            {
                "title": "Reproducible local transformation",
                "detail": "Raw tables were profiled, converted to one row per accepted order, and validated with executable notebooks and assertions.",
            },
        ],
        "methods": methods or [],
    }


def main() -> None:
    metric_profile = json.loads(
        (OUT / "metrics" / "metric_diagnostic_profile.json").read_text(encoding="utf-8")
    )
    order_profile = json.loads((OUT / "order_mart_profile.json").read_text(encoding="utf-8"))
    capacity_profile = json.loads(
        (OUT / "capacity" / "capacity_profile.json").read_text(encoding="utf-8")
    )

    daily = pd.read_csv(OUT / "metrics" / "daily_kpi.csv")
    meal = pd.read_csv(OUT / "metrics" / "meal_period_kpi.csv")
    district = pd.read_csv(OUT / "metrics" / "district_kpi.csv").head(10)
    stage = pd.read_csv(OUT / "metrics" / "late_vs_on_time_stage_gap.csv")
    model = pd.read_csv(OUT / "model" / "model_metrics.csv")
    capacity = pd.read_csv(OUT / "capacity" / "scenario_summary.csv")

    executive_rows = [
        {
            "strategyOrders": metric_profile["strategy_orders"],
            "lateOrders": metric_profile["strategy_late_orders"],
            "lateRate": metric_profile["strategy_late_rate_pct"] / 100,
            "peakOrderShare": metric_profile["peak_order_share_pct"] / 100,
            "peakLateOrderShare": metric_profile["peak_late_order_share_pct"] / 100,
            "topFiveDistrictLateShare": sum(
                row["late_order_contribution_pct"]
                for row in metric_profile["top_5_districts_by_late_orders"]
            )
            / 100,
            "topFiveDistrictOrderShare": metric_profile["top_5_district_order_share_pct"] / 100,
            "testAcceptanceModelPrAuc": float(
                model.loc[
                    model["split"].eq("test_20221023_20221024")
                    & model["model"].eq("acceptance_time_model"),
                    "pr_auc",
                ].iloc[0]
            ),
            "testAcceptanceModelLiftTop10": float(
                model.loc[
                    model["split"].eq("test_20221023_20221024")
                    & model["model"].eq("acceptance_time_model"),
                    "lift_at_top_10pct",
                ].iloc[0]
            ),
        }
    ]

    scope_rows = [
        {
            "scope": "All valid completed orders",
            "orders": order_profile["valid_completed_orders"],
            "onTimeRate": order_profile["overall_on_time_rate_pct"] / 100,
        },
        {
            "scope": "Strategy eligible: on-demand and known district",
            "orders": order_profile["strategy_eligible_orders"],
            "onTimeRate": order_profile["strategy_sample_on_time_rate_pct"] / 100,
        },
    ]

    snapshot = {
        "surface": "report",
        "title": "超时集中在高峰与少数商圈，运力配置应先做风险优先试点",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "asOf": "2022-10-24",
        "status": "reviewed-public-data-analysis",
        "buildStatus": "complete",
        "filters": [],
        "queries": {
            "executive_summary": {
                "rows": executive_rows,
                "source": source(
                    "Reviewed project summary from Meituan public challenge data",
                    ["Official four-table public dataset", "order_fulfillment_mart.parquet"],
                    ["Strategy sample: valid completed, on-demand, known district"],
                    [
                        {
                            "label": "Late rate",
                            "definition": "Orders delivered after the estimated arrival timestamp divided by valid completed orders.",
                            "componentIds": ["report-executive-summary"],
                        },
                        {
                            "label": "Peak late-order share",
                            "definition": "Share of strategy-sample late orders entering delivery during [10:30, 14:00) or [17:00, 21:00).",
                            "componentIds": ["report-executive-summary", "report-peak-concentration"],
                        },
                    ],
                ),
            },
            "sample_scope": {
                "rows": scope_rows,
                "source": source(
                    "Order-mart sample audit",
                    ["order_fulfillment_mart.parquet", "order_mart_profile.json"],
                    ["One accepted waybill per order"],
                    [
                        {
                            "label": "On-time rate",
                            "definition": "Orders with actual arrival no later than ETA divided by valid completed orders in the stated scope.",
                            "componentIds": ["report-methods"],
                        }
                    ],
                ),
            },
            "daily_kpi": {
                "rows": records(daily),
                "source": source(
                    "Daily strategy-sample KPIs",
                    ["daily_kpi.csv"],
                    ["2022-10-17 through 2022-10-24"],
                    [
                        {
                            "label": "Daily late rate",
                            "definition": "Late orders divided by strategy-eligible orders for each business date.",
                            "componentIds": ["report-peak-concentration"],
                        }
                    ],
                ),
            },
            "meal_period_kpi": {
                "rows": records(meal),
                "source": source(
                    "Meal-period fulfillment KPIs",
                    ["meal_period_kpi.csv"],
                    ["Lunch: [10:30, 14:00); dinner: [17:00, 21:00)"],
                    [
                        {
                            "label": "Meal-period late rate",
                            "definition": "Late orders divided by strategy orders entering delivery in the stated meal period.",
                            "componentIds": ["report-peak-concentration"],
                        }
                    ],
                ),
            },
            "district_kpi": {
                "rows": records(district),
                "source": source(
                    "Top anonymous districts by late-order count",
                    ["district_kpi.csv"],
                    ["Top 10 of 22 known anonymous districts"],
                    [
                        {
                            "label": "Late-order contribution",
                            "definition": "District late orders divided by all strategy-sample late orders.",
                            "componentIds": ["report-district-concentration"],
                        }
                    ],
                ),
            },
            "stage_gap": {
                "rows": records(stage),
                "source": source(
                    "Late versus on-time stage-duration difference",
                    ["late_vs_on_time_stage_gap.csv"],
                    ["Strategy-eligible orders"],
                    [
                        {
                            "label": "Mean stage-duration gap",
                            "definition": "Mean stage minutes among late orders minus mean stage minutes among on-time orders; descriptive, not causal.",
                            "componentIds": ["report-stage-decomposition"],
                        }
                    ],
                ),
            },
            "model_metrics": {
                "rows": records(model),
                "source": source(
                    "Temporal holdout late-risk model evaluation",
                    ["model_metrics.csv"],
                    ["Final test: 2022-10-23 through 2022-10-24"],
                    [
                        {
                            "label": "Recall at top 10%",
                            "definition": "Share of actual late orders captured in the highest predicted-risk 10% of orders.",
                            "componentIds": ["report-model-performance"],
                        },
                        {
                            "label": "Lift at top 10%",
                            "definition": "Late rate among the highest-risk 10% divided by the overall late rate in the same evaluation set.",
                            "componentIds": ["report-model-performance"],
                        },
                    ],
                ),
            },
            "capacity_scenario": {
                "rows": records(capacity),
                "source": source(
                    "Six temporal-holdout noon-peak capacity scenarios",
                    ["scenario_summary.csv", "snapshot_grid_metrics.parquet"],
                    [
                        "Evaluation dates: 2022-10-23 and 2022-10-24",
                        "500m x 400m grids; candidate supply within 2km",
                        "Budgets: 10, 20, 30, 50 rider slots per epoch",
                    ],
                    [
                        {
                            "label": "Hindsight actual-late coverage",
                            "definition": "Share of actual late orders located in grids selected by the pre-decision rule; evaluation only, not an estimated reduction.",
                            "componentIds": ["report-capacity-scenario"],
                        },
                        {
                            "label": "Rider slot",
                            "definition": "One hypothetical incremental candidate-rider position at a dispatch epoch; not a rider-hour or confirmed staffing budget.",
                            "componentIds": ["report-capacity-scenario"],
                        },
                    ],
                ),
            },
        },
        "meta": {
            "dataset": "Meituan-INFORMS-TSL Research Challenge public data",
            "dataWindow": "2022-10-17 to 2022-10-24",
            "analysisType": "Independent offline analysis based on Meituan's official public research dataset",
            "capacityEvaluation": {
                "epochs": capacity_profile["epochs"],
                "waitingOrders": capacity_profile["waiting_orders"],
                "modelScoredShare": capacity_profile["model_scored_order_share_pct"] / 100,
            },
        },
    }

    SNAPSHOT_PATH.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    print(SNAPSHOT_PATH)


if __name__ == "__main__":
    main()
