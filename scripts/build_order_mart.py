from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "downloads"
OUTPUT = ROOT / "data" / "processed" / "order_fulfillment_mart.parquet"
PROFILE_OUTPUT = ROOT / "outputs" / "order_mart_profile.json"


def unix_to_shanghai(series: pd.Series) -> pd.Series:
    """Convert positive Unix seconds to timezone-naive Asia/Shanghai datetime."""
    valid = series.where(series.gt(0))
    return pd.to_datetime(valid, unit="s", utc=True).dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized great-circle distance for shifted but internally consistent coordinates."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(a))


def minutes_between(end: pd.Series, start: pd.Series) -> pd.Series:
    return (end - start) / 60.0


def parse_order_list(value) -> list[int]:
    parsed = ast.literal_eval(value) if isinstance(value, str) else value
    return [int(item) for item in parsed]


def main() -> None:
    waybill_path = RAW / "waybill_extracted" / "all_waybill_info_meituan_0322.csv"
    wave_path = RAW / "courier_wave_info_meituan.csv"

    waybill = pd.read_csv(waybill_path).rename(columns={"Unnamed: 0": "source_row_id"})
    wave = pd.read_csv(wave_path)

    assert waybill["waybill_id"].is_unique, "waybill_id must be unique"

    # Zero is a missing-value sentinel in event-time columns, so replace it
    # before calculating order-level first-event timestamps.
    attempt_source = waybill.assign(
        initial_platform_order_time=waybill["platform_order_time"].where(
            waybill["platform_order_time"].gt(0)
        ),
        initial_order_push_time=waybill["order_push_time"].where(waybill["order_push_time"].gt(0)),
        positive_dispatch_time=waybill["dispatch_time"].where(waybill["dispatch_time"].gt(0)),
    )
    attempt_agg = (
        attempt_source.groupby("order_id", observed=True)
        .agg(
            waybill_attempt_count=("waybill_id", "size"),
            accepted_waybill_count=("is_courier_grabbed", "sum"),
            platform_order_time=("initial_platform_order_time", "min"),
            order_push_time=("initial_order_push_time", "min"),
            first_dispatch_time=("positive_dispatch_time", "min"),
            last_dispatch_time=("dispatch_time", "max"),
        )
        .reset_index()
    )
    attempt_agg["rejected_waybill_count"] = (
        attempt_agg["waybill_attempt_count"] - attempt_agg["accepted_waybill_count"]
    )
    for column in ["platform_order_time", "order_push_time", "first_dispatch_time"]:
        attempt_agg[column] = attempt_agg[column].fillna(0).astype("int64")

    orders = (
        waybill.loc[waybill["is_courier_grabbed"].eq(1)]
        .rename(
            columns={
                "platform_order_time": "accepted_waybill_platform_order_time",
                "order_push_time": "accepted_waybill_order_push_time",
            }
        )
        .copy()
    )
    assert orders["order_id"].is_unique, "Each order must have exactly one accepted waybill"
    assert len(orders) == waybill["order_id"].nunique(), "Every order must have one accepted waybill"

    orders = orders.merge(attempt_agg, on="order_id", how="left", validate="one_to_one")
    assert orders["dispatch_time"].eq(orders["last_dispatch_time"]).all(), (
        "Accepted waybill should be the final dispatch attempt"
    )

    parsed_order_ids = wave["order_ids"].map(parse_order_list)
    wave = wave.assign(
        wave_order_count=parsed_order_ids.map(len),
        order_id=parsed_order_ids,
    ).explode("order_id", ignore_index=True)
    wave["order_id"] = wave["order_id"].astype("int64")
    wave_map = wave[
        [
            "order_id",
            "courier_id",
            "wave_id",
            "wave_start_time",
            "wave_end_time",
            "wave_order_count",
        ]
    ].rename(columns={"courier_id": "wave_courier_id"})
    assert wave_map["order_id"].is_unique, "Accepted order should belong to at most one courier wave"
    orders = orders.merge(wave_map, on="order_id", how="left", validate="one_to_one")
    wave_matched = orders["wave_courier_id"].notna()
    assert orders.loc[wave_matched, "courier_id"].eq(
        orders.loc[wave_matched, "wave_courier_id"]
    ).all(), "Wave courier must match the accepted-waybill courier"
    orders = orders.drop(columns="wave_courier_id")

    timestamp_columns = [
        "platform_order_time",
        "order_push_time",
        "accepted_waybill_platform_order_time",
        "accepted_waybill_order_push_time",
        "first_dispatch_time",
        "dispatch_time",
        "grab_time",
        "fetch_time",
        "arrive_time",
        "estimate_arrived_time",
        "estimate_meal_prepare_time",
        "wave_start_time",
        "wave_end_time",
    ]
    for column in timestamp_columns:
        orders[f"{column}_local"] = unix_to_shanghai(orders[column])

    orders["service_date"] = pd.to_datetime(orders["dt"].astype(str), format="%Y%m%d")
    orders["push_time_slot_30m"] = orders["order_push_time_local"].dt.floor("30min")
    orders["push_hour"] = orders["order_push_time_local"].dt.hour
    orders["push_weekday"] = orders["order_push_time_local"].dt.dayofweek

    orders["order_to_push_minutes"] = minutes_between(
        orders["order_push_time"], orders["platform_order_time"]
    )
    orders["first_dispatch_wait_minutes"] = minutes_between(
        orders["first_dispatch_time"], orders["order_push_time"]
    ).where(orders["first_dispatch_time"].gt(0))
    orders["accepted_dispatch_wait_minutes"] = minutes_between(
        orders["dispatch_time"], orders["order_push_time"]
    ).where(orders["dispatch_time"].gt(0))
    orders["accepted_dispatch_to_grab_minutes"] = minutes_between(
        orders["grab_time"], orders["dispatch_time"]
    ).where(orders["dispatch_time"].gt(0))
    orders["order_push_to_accept_minutes"] = minutes_between(
        orders["grab_time"], orders["order_push_time"]
    )
    orders["accept_to_pickup_minutes"] = minutes_between(orders["fetch_time"], orders["grab_time"])
    orders["pickup_to_delivery_minutes"] = minutes_between(orders["arrive_time"], orders["fetch_time"])
    orders["total_fulfillment_minutes"] = minutes_between(
        orders["arrive_time"], orders["platform_order_time"]
    )
    orders["promise_minutes"] = minutes_between(
        orders["estimate_arrived_time"], orders["platform_order_time"]
    )
    orders["signed_late_minutes"] = minutes_between(
        orders["arrive_time"], orders["estimate_arrived_time"]
    )
    orders["late_minutes"] = orders["signed_late_minutes"].clip(lower=0)

    chronology_valid = (
        orders["platform_order_time"].le(orders["order_push_time"])
        & orders["order_push_time"].le(orders["grab_time"])
        & orders["grab_time"].le(orders["fetch_time"])
        & orders["fetch_time"].le(orders["arrive_time"])
    )
    orders["is_dispatch_timeline_valid"] = (
        orders["first_dispatch_time"].gt(0)
        & orders["dispatch_time"].gt(0)
        & orders["order_push_time"].le(orders["first_dispatch_time"])
        & orders["first_dispatch_time"].le(orders["dispatch_time"])
        & orders["dispatch_time"].le(orders["grab_time"])
    )
    orders["is_valid_completed"] = (
        orders["arrive_time"].gt(0)
        & orders["estimate_arrived_time"].gt(0)
        & chronology_valid
    )
    orders["is_on_time"] = pd.array([pd.NA] * len(orders), dtype="boolean")
    valid = orders["is_valid_completed"]
    orders.loc[valid, "is_on_time"] = orders.loc[valid, "arrive_time"].le(
        orders.loc[valid, "estimate_arrived_time"]
    )
    orders["is_late"] = (~orders["is_on_time"]).astype("boolean")
    orders.loc[~valid, "is_late"] = pd.NA

    orders["late_severity"] = pd.cut(
        orders["signed_late_minutes"],
        bins=[-np.inf, 0, 10, 30, np.inf],
        labels=["on_time", "late_0_10m", "late_10_30m", "late_30m_plus"],
        right=True,
    )
    orders.loc[~valid, "late_severity"] = pd.NA

    scale = 1_000_000.0
    orders["merchant_customer_distance_km"] = haversine_km(
        orders["sender_lat"] / scale,
        orders["sender_lng"] / scale,
        orders["recipient_lat"] / scale,
        orders["recipient_lng"] / scale,
    )
    orders["courier_merchant_distance_at_accept_km"] = haversine_km(
        orders["grab_lat"] / scale,
        orders["grab_lng"] / scale,
        orders["sender_lat"] / scale,
        orders["sender_lng"] / scale,
    )

    orders["is_on_demand"] = orders["is_prebook"].eq(0)
    orders["is_known_da"] = orders["da_id"].ne(0)
    orders["is_strategy_eligible"] = (
        orders["is_valid_completed"] & orders["is_on_demand"] & orders["is_known_da"]
    )

    stage_columns = [
        "first_dispatch_wait_minutes",
        "accepted_dispatch_wait_minutes",
        "accepted_dispatch_to_grab_minutes",
        "order_push_to_accept_minutes",
        "accept_to_pickup_minutes",
        "pickup_to_delivery_minutes",
    ]
    assert not orders.loc[orders["is_valid_completed"], stage_columns].lt(0).any().any(), (
        "Valid completed orders cannot have negative stage duration"
    )
    assert int(orders["is_valid_completed"].sum()) == 568_545
    assert int(orders["wave_id"].notna().sum()) == 568_545

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    PROFILE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    orders.to_parquet(OUTPUT, index=False, compression="zstd")

    valid_orders = orders.loc[orders["is_valid_completed"]]
    strategy_orders = orders.loc[orders["is_strategy_eligible"]]
    profile = {
        "output_path": str(OUTPUT.relative_to(ROOT)),
        "rows": int(len(orders)),
        "columns": int(orders.shape[1]),
        "unique_orders": int(orders["order_id"].nunique()),
        "valid_completed_orders": int(orders["is_valid_completed"].sum()),
        "on_demand_orders": int(orders["is_on_demand"].sum()),
        "known_da_orders": int(orders["is_known_da"].sum()),
        "strategy_eligible_orders": int(orders["is_strategy_eligible"].sum()),
        "overall_on_time_rate_pct": round(float(valid_orders["is_on_time"].mean() * 100), 4),
        "strategy_sample_on_time_rate_pct": round(
            float(strategy_orders["is_on_time"].mean() * 100), 4
        ),
        "wave_match_rows": int(orders["wave_id"].notna().sum()),
        "valid_dispatch_timeline_orders": int(orders["is_dispatch_timeline_valid"].sum()),
        "service_date_min": orders["service_date"].min().date().isoformat(),
        "service_date_max": orders["service_date"].max().date().isoformat(),
        "parquet_bytes": int(OUTPUT.stat().st_size),
        "late_severity_counts": {
            str(k): int(v) for k, v in valid_orders["late_severity"].value_counts().items()
        },
    }
    PROFILE_OUTPUT.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(profile, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
