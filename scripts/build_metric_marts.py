from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MART_PATH = ROOT / "data" / "processed" / "order_fulfillment_mart.parquet"
OUTPUT_DIR = ROOT / "outputs" / "metrics"


def add_rates(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["late_rate"] = frame["late_orders"] / frame["orders"]
    frame["on_time_rate"] = 1 - frame["late_rate"]
    return frame


def meal_period(slot: pd.Series) -> pd.Categorical:
    hour = slot.dt.hour + slot.dt.minute / 60
    return pd.cut(
        hour,
        bins=[-1, 10.5, 14, 17, 21, 24],
        labels=["off_morning", "lunch", "off_afternoon", "dinner", "off_night"],
        right=False,
    )


def aggregate_kpis(data: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    result = (
        data.groupby(group_columns, observed=True)
        .agg(
            orders=("order_id", "size"),
            late_orders=("is_late", "sum"),
            late_minutes_sum=("late_minutes", "sum"),
            accepting_couriers=("courier_id", "nunique"),
            avg_order_to_push_minutes=("order_to_push_minutes", "mean"),
            avg_order_push_to_accept_minutes=("order_push_to_accept_minutes", "mean"),
            avg_accept_to_pickup_minutes=("accept_to_pickup_minutes", "mean"),
            avg_pickup_to_delivery_minutes=("pickup_to_delivery_minutes", "mean"),
            avg_total_fulfillment_minutes=("total_fulfillment_minutes", "mean"),
            avg_promise_minutes=("promise_minutes", "mean"),
            avg_merchant_customer_distance_km=("merchant_customer_distance_km", "mean"),
            avg_waybill_attempt_count=("waybill_attempt_count", "mean"),
        )
        .reset_index()
    )
    result = add_rates(result)
    result["avg_late_minutes_per_late_order"] = (
        result["late_minutes_sum"] / result["late_orders"].where(result["late_orders"].gt(0))
    )
    # This is an activity proxy, not available supply: only couriers accepting
    # at least one order in the cell appear in the denominator.
    result["orders_per_accepting_courier_proxy"] = result["orders"] / result["accepting_couriers"]
    return result


def main() -> None:
    orders = pd.read_parquet(MART_PATH)
    strategy = orders.loc[orders["is_strategy_eligible"]].copy()
    strategy["meal_period"] = meal_period(strategy["push_time_slot_30m"])
    strategy["is_meal_peak"] = strategy["meal_period"].isin(["lunch", "dinner"])
    strategy["time_of_day"] = strategy["push_time_slot_30m"].dt.strftime("%H:%M")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    daily = aggregate_kpis(strategy, ["dt"])
    time_of_day = aggregate_kpis(strategy, ["time_of_day", "meal_period"])
    district = aggregate_kpis(strategy, ["da_id"])
    district["late_order_contribution"] = district["late_orders"] / district["late_orders"].sum()
    district = district.sort_values("late_orders", ascending=False)
    meal_period_summary = aggregate_kpis(strategy, ["meal_period"])
    district_meal_period = aggregate_kpis(strategy, ["da_id", "meal_period"])
    cell = aggregate_kpis(strategy, ["dt", "da_id", "push_time_slot_30m", "meal_period"])

    stage_columns = [
        "order_to_push_minutes",
        "order_push_to_accept_minutes",
        "accept_to_pickup_minutes",
        "pickup_to_delivery_minutes",
    ]
    stage_rows = []
    total_gap = (
        strategy.loc[strategy["is_late"], "total_fulfillment_minutes"].mean()
        - strategy.loc[~strategy["is_late"], "total_fulfillment_minutes"].mean()
    )
    for column in stage_columns:
        on_time = strategy.loc[~strategy["is_late"], column]
        late = strategy.loc[strategy["is_late"], column]
        mean_gap = late.mean() - on_time.mean()
        stage_rows.append(
            {
                "stage": column,
                "mean_on_time": on_time.mean(),
                "mean_late": late.mean(),
                "mean_gap_minutes": mean_gap,
                "gap_share": mean_gap / total_gap,
                "p50_on_time": on_time.median(),
                "p50_late": late.median(),
                "p90_on_time": on_time.quantile(0.9),
                "p90_late": late.quantile(0.9),
            }
        )
    stage_gap = pd.DataFrame(stage_rows)
    late_only = strategy.loc[strategy["is_late"]].copy()
    late_severity_share = (
        late_only["late_severity"].value_counts(normalize=True) * 100
    )

    assert abs(stage_gap["mean_gap_minutes"].sum() - total_gap) < 1e-9
    assert int(daily["orders"].sum()) == len(strategy)
    assert int(daily["late_orders"].sum()) == int(strategy["is_late"].sum())
    assert int(meal_period_summary["orders"].sum()) == len(strategy)

    daily.to_csv(OUTPUT_DIR / "daily_kpi.csv", index=False, encoding="utf-8-sig")
    time_of_day.to_csv(OUTPUT_DIR / "time_of_day_kpi.csv", index=False, encoding="utf-8-sig")
    district.to_csv(OUTPUT_DIR / "district_kpi.csv", index=False, encoding="utf-8-sig")
    meal_period_summary.to_csv(
        OUTPUT_DIR / "meal_period_kpi.csv", index=False, encoding="utf-8-sig"
    )
    district_meal_period.to_csv(
        OUTPUT_DIR / "district_meal_period_kpi.csv", index=False, encoding="utf-8-sig"
    )
    cell.to_parquet(OUTPUT_DIR / "district_date_slot_kpi.parquet", index=False, compression="zstd")
    stage_gap.to_csv(OUTPUT_DIR / "late_vs_on_time_stage_gap.csv", index=False, encoding="utf-8-sig")

    peak = meal_period_summary.loc[meal_period_summary["meal_period"].isin(["lunch", "dinner"])]
    profile = {
        "strategy_orders": int(len(strategy)),
        "strategy_late_orders": int(strategy["is_late"].sum()),
        "strategy_late_rate_pct": round(float(strategy["is_late"].mean() * 100), 4),
        "peak_orders": int(peak["orders"].sum()),
        "peak_order_share_pct": round(float(peak["orders"].sum() / len(strategy) * 100), 4),
        "peak_late_orders": int(peak["late_orders"].sum()),
        "peak_late_order_share_pct": round(
            float(peak["late_orders"].sum() / strategy["is_late"].sum() * 100), 4
        ),
        "peak_late_rate_pct": round(float(peak["late_orders"].sum() / peak["orders"].sum() * 100), 4),
        "off_peak_late_rate_pct": round(
            float(
                (strategy["is_late"].sum() - peak["late_orders"].sum())
                / (len(strategy) - peak["orders"].sum())
                * 100
            ),
            4,
        ),
        "top_5_district_order_share_pct": round(
            float(district.head(5)["orders"].sum() / len(strategy) * 100), 4
        ),
        "late_vs_on_time_total_duration_gap_minutes": round(float(total_gap), 4),
        "late_minutes_summary": {
            "mean": round(float(late_only["late_minutes"].mean()), 4),
            "p50": round(float(late_only["late_minutes"].median()), 4),
            "p90": round(float(late_only["late_minutes"].quantile(0.9)), 4),
        },
        "late_severity_share_pct_among_late_orders": {
            str(level): round(float(share), 4)
            for level, share in late_severity_share.items()
            if str(level) != "on_time"
        },
        "stage_gap_minutes": {
            row["stage"]: round(float(row["mean_gap_minutes"]), 4)
            for _, row in stage_gap.iterrows()
        },
        "stage_gap_share_pct": {
            row["stage"]: round(float(row["gap_share"] * 100), 4)
            for _, row in stage_gap.iterrows()
        },
        "top_5_districts_by_late_orders": [
            {
                "da_id": int(row["da_id"]),
                "orders": int(row["orders"]),
                "late_orders": int(row["late_orders"]),
                "late_rate_pct": round(float(row["late_rate"] * 100), 4),
                "late_order_contribution_pct": round(
                    float(row["late_order_contribution"] * 100), 4
                ),
            }
            for _, row in district.head(5).iterrows()
        ],
    }
    (OUTPUT_DIR / "metric_diagnostic_profile.json").write_text(
        json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(profile, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
