from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "downloads"
MART_PATH = ROOT / "data" / "processed" / "order_fulfillment_mart.parquet"
MODEL_PATH = ROOT / "models" / "push_time_late_risk.joblib"
OUTPUT_DIR = ROOT / "outputs" / "capacity"

GRID_X_METERS = 500.0
GRID_Y_METERS = 400.0
SUPPLY_RADIUS_METERS = 2_000.0
SMOOTHING_STRENGTH = 50.0
BUDGETS = [10, 20, 30, 50]


def parse_load(value: str) -> int:
    parsed = ast.literal_eval(value) if isinstance(value, str) else value
    return len(parsed)


def project_coordinates(lat_scaled, lng_scaled, mean_lat_radians):
    lat = np.asarray(lat_scaled, dtype=float) / 1_000_000.0
    lng = np.asarray(lng_scaled, dtype=float) / 1_000_000.0
    x = lng * 111_320.0 * np.cos(mean_lat_radians)
    y = lat * 110_540.0
    return x, y


def greedy_allocate(frame: pd.DataFrame, budget: int, numerator: str) -> np.ndarray:
    allocated = np.zeros(len(frame), dtype=int)
    base_supply = frame["candidate_riders_within_2km"].to_numpy(dtype=float)
    demand_signal = frame[numerator].to_numpy(dtype=float)
    max_per_grid = max(1, math.ceil(budget * 0.25))
    for _ in range(budget):
        marginal_score = demand_signal / (base_supply + allocated + 1.0)
        marginal_score[allocated >= max_per_grid] = -np.inf
        chosen = int(np.argmax(marginal_score))
        allocated[chosen] += 1
    return allocated


def main() -> None:
    model_bundle = joblib.load(MODEL_PATH)
    model_features = model_bundle["features"]
    orders = pd.read_parquet(
        MART_PATH,
        columns=[
            "order_id",
            "dt",
            "sender_lat",
            "sender_lng",
            "is_valid_completed",
            "is_late",
            "is_strategy_eligible",
        ]
        + model_features,
    )
    dispatch_orders = pd.read_csv(RAW / "dispatch_waybill_meituan.csv").rename(
        columns={"Unnamed: 0": "source_row_id"}
    )
    riders = pd.read_csv(RAW / "dispatch_rider_meituan.csv").rename(
        columns={"Unnamed: 0": "source_row_id"}
    )
    riders["on_hand_order_count"] = riders["courier_waybills"].map(parse_load)
    riders["is_idle_candidate"] = riders["on_hand_order_count"].eq(0)
    assert not riders.duplicated(["dt", "dispatch_time", "courier_id"]).any(), (
        "Candidate courier must be unique within each dispatch snapshot"
    )

    # Use one fixed projection and grid origin for historical orders, current
    # waiting orders, and candidate riders.
    mean_lat_radians = np.radians(
        pd.concat(
            [orders["sender_lat"], riders["rider_lat"]], ignore_index=True
        ).median()
        / 1_000_000.0
    )
    order_x, order_y = project_coordinates(
        orders["sender_lat"], orders["sender_lng"], mean_lat_radians
    )
    rider_x, rider_y = project_coordinates(
        riders["rider_lat"], riders["rider_lng"], mean_lat_radians
    )
    x_origin = np.floor(min(order_x.min(), rider_x.min()) / GRID_X_METERS) * GRID_X_METERS
    y_origin = np.floor(min(order_y.min(), rider_y.min()) / GRID_Y_METERS) * GRID_Y_METERS

    orders["x_m"] = order_x
    orders["y_m"] = order_y
    orders["grid_x"] = np.floor((orders["x_m"] - x_origin) / GRID_X_METERS).astype(int)
    orders["grid_y"] = np.floor((orders["y_m"] - y_origin) / GRID_Y_METERS).astype(int)
    riders["x_m"] = rider_x
    riders["y_m"] = rider_y

    waiting = dispatch_orders.merge(
        orders,
        on="order_id",
        how="left",
        validate="one_to_one",
        suffixes=("_snapshot", ""),
    )
    assert waiting["is_valid_completed"].all()
    assert waiting["order_id"].is_unique

    # Temporal holdout: allocate only on the final two dates, using historical
    # outcomes strictly before each snapshot date.
    waiting = waiting.loc[waiting["dt_snapshot"].ge(20221023)].copy()
    waiting["model_risk"] = np.nan
    score_mask = waiting["is_strategy_eligible"]
    scoring_features = waiting.loc[score_mask, model_features].copy()
    for column, levels in model_bundle["categories"].items():
        known = scoring_features[column].where(scoring_features[column].isin(levels))
        scoring_features[column] = pd.Categorical(known, categories=levels)
    waiting.loc[score_mask, "model_risk"] = model_bundle["model"].predict_proba(
        scoring_features
    )[:, 1]
    rider_eval = riders.loc[riders["dt"].ge(20221023)].copy()
    grid_frames = []

    for (dt, epoch), current_orders in waiting.groupby(
        ["dt_snapshot", "dispatch_time"], sort=True
    ):
        current_riders = rider_eval.loc[
            rider_eval["dt"].eq(dt) & rider_eval["dispatch_time"].eq(epoch)
        ].copy()
        assert len(current_riders) > 0

        history = orders.loc[orders["is_valid_completed"] & orders["dt"].lt(dt)]
        global_prior = float(history["is_late"].mean())
        historical_grid = (
            history.groupby(["grid_x", "grid_y"], observed=True)
            .agg(historical_orders=("order_id", "size"), historical_late_orders=("is_late", "sum"))
            .reset_index()
        )
        historical_grid["smoothed_historical_late_rate"] = (
            historical_grid["historical_late_orders"] + SMOOTHING_STRENGTH * global_prior
        ) / (historical_grid["historical_orders"] + SMOOTHING_STRENGTH)

        grid = (
            current_orders.groupby(["grid_x", "grid_y"], observed=True)
            .agg(
                waiting_orders=("order_id", "size"),
                actual_late_orders=("is_late", "sum"),
                model_scored_orders=("model_risk", "count"),
                model_risk_sum=("model_risk", "sum"),
            )
            .reset_index()
        )
        grid = grid.merge(historical_grid, on=["grid_x", "grid_y"], how="left")
        grid["historical_orders"] = grid["historical_orders"].fillna(0).astype(int)
        grid["historical_late_orders"] = grid["historical_late_orders"].fillna(0).astype(int)
        grid["smoothed_historical_late_rate"] = grid[
            "smoothed_historical_late_rate"
        ].fillna(global_prior)
        grid["fallback_orders"] = grid["waiting_orders"] - grid["model_scored_orders"]
        grid["expected_late_demand"] = grid["model_risk_sum"] + (
            grid["fallback_orders"] * grid["smoothed_historical_late_rate"]
        )
        grid["expected_late_rate"] = grid["expected_late_demand"] / grid["waiting_orders"]
        grid["actual_late_rate"] = grid["actual_late_orders"] / grid["waiting_orders"]
        grid["grid_center_x_m"] = x_origin + (grid["grid_x"] + 0.5) * GRID_X_METERS
        grid["grid_center_y_m"] = y_origin + (grid["grid_y"] + 0.5) * GRID_Y_METERS

        rider_tree = cKDTree(current_riders[["x_m", "y_m"]].to_numpy())
        centers = grid[["grid_center_x_m", "grid_center_y_m"]].to_numpy()
        grid["candidate_riders_within_2km"] = rider_tree.query_ball_point(
            centers, r=SUPPLY_RADIUS_METERS, return_length=True
        )
        idle = current_riders.loc[current_riders["is_idle_candidate"]]
        if len(idle):
            idle_tree = cKDTree(idle[["x_m", "y_m"]].to_numpy())
            grid["idle_candidate_riders_within_2km"] = idle_tree.query_ball_point(
                centers, r=SUPPLY_RADIUS_METERS, return_length=True
            )
        else:
            grid["idle_candidate_riders_within_2km"] = 0

        grid["waiting_orders_per_candidate_2km"] = grid["waiting_orders"] / grid[
            "candidate_riders_within_2km"
        ].clip(lower=1)
        grid["risk_pressure"] = grid["expected_late_demand"] / (
            grid["candidate_riders_within_2km"] + 1
        )
        grid["dt"] = int(dt)
        grid["dispatch_time"] = int(epoch)
        grid_frames.append(grid)

    snapshot_grid = pd.concat(grid_frames, ignore_index=True)
    snapshot_grid["dispatch_time_local"] = (
        pd.to_datetime(snapshot_grid["dispatch_time"], unit="s", utc=True)
        .dt.tz_convert("Asia/Shanghai")
        .dt.tz_localize(None)
    )

    allocation_rows = []
    for (dt, epoch), epoch_grid in snapshot_grid.groupby(["dt", "dispatch_time"], sort=True):
        epoch_grid = epoch_grid.reset_index(drop=True)
        for budget in BUDGETS:
            for policy, numerator in [
                ("risk_adjusted", "expected_late_demand"),
                ("volume_only", "waiting_orders"),
            ]:
                allocation = greedy_allocate(epoch_grid, budget, numerator)
                chosen_mask = allocation > 0
                chosen = epoch_grid.loc[chosen_mask].copy()
                chosen["allocated_rider_slots"] = allocation[chosen_mask]
                chosen["policy"] = policy
                chosen["budget_rider_slots"] = budget
                allocation_rows.append(chosen)

    allocations = pd.concat(allocation_rows, ignore_index=True)

    snapshot_policy_rows = []
    for (dt, epoch, budget, policy), chosen in allocations.groupby(
        ["dt", "dispatch_time", "budget_rider_slots", "policy"], sort=True
    ):
        all_grid = snapshot_grid.loc[
            snapshot_grid["dt"].eq(dt) & snapshot_grid["dispatch_time"].eq(epoch)
        ]
        total_waiting = int(all_grid["waiting_orders"].sum())
        total_expected = float(all_grid["expected_late_demand"].sum())
        total_actual_late = int(all_grid["actual_late_orders"].sum())
        allocated = int(chosen["allocated_rider_slots"].sum())
        snapshot_policy_rows.append(
            {
                "dt": int(dt),
                "dispatch_time": int(epoch),
                "dispatch_time_local": chosen["dispatch_time_local"].iloc[0],
                "budget_rider_slots": int(budget),
                "policy": policy,
                "allocated_rider_slots": allocated,
                "targeted_grid_count": int(len(chosen)),
                "waiting_orders": total_waiting,
                "actual_late_orders": total_actual_late,
                "waiting_order_coverage_pct": float(
                    chosen["waiting_orders"].sum() / total_waiting * 100
                ),
                "expected_late_risk_coverage_pct": float(
                    chosen["expected_late_demand"].sum() / total_expected * 100
                ),
                "actual_late_order_coverage_pct_hindsight": float(
                    chosen["actual_late_orders"].sum() / total_actual_late * 100
                ),
                "max_single_grid_allocation_share_pct": float(
                    chosen["allocated_rider_slots"].max() / allocated * 100
                ),
            }
        )
    snapshot_policy_metrics = pd.DataFrame(snapshot_policy_rows).sort_values(
        ["budget_rider_slots", "dt", "dispatch_time", "policy"]
    )

    summary_rows = []
    for (budget, policy), group in allocations.groupby(
        ["budget_rider_slots", "policy"], sort=True
    ):
        total_waiting = 0
        total_expected = 0.0
        total_actual_late = 0
        covered_waiting = 0
        covered_expected = 0.0
        covered_actual_late = 0
        weighted_actual_rate_numerator = 0.0
        weighted_expected_rate_numerator = 0.0
        total_allocated = 0
        max_concentration = []
        targeted_grid_count = 0
        for (dt, epoch), chosen in group.groupby(["dt", "dispatch_time"]):
            all_grid = snapshot_grid.loc[
                snapshot_grid["dt"].eq(dt) & snapshot_grid["dispatch_time"].eq(epoch)
            ]
            total_waiting += int(all_grid["waiting_orders"].sum())
            total_expected += float(all_grid["expected_late_demand"].sum())
            total_actual_late += int(all_grid["actual_late_orders"].sum())
            covered_waiting += int(chosen["waiting_orders"].sum())
            covered_expected += float(chosen["expected_late_demand"].sum())
            covered_actual_late += int(chosen["actual_late_orders"].sum())
            weighted_actual_rate_numerator += float(
                (chosen["allocated_rider_slots"] * chosen["actual_late_rate"]).sum()
            )
            weighted_expected_rate_numerator += float(
                (chosen["allocated_rider_slots"] * chosen["expected_late_rate"]).sum()
            )
            allocated = int(chosen["allocated_rider_slots"].sum())
            total_allocated += allocated
            max_concentration.append(float(chosen["allocated_rider_slots"].max() / allocated))
            targeted_grid_count += len(chosen)

        summary_rows.append(
            {
                "budget_rider_slots_per_epoch": int(budget),
                "policy": policy,
                "epochs": int(group[["dt", "dispatch_time"]].drop_duplicates().shape[0]),
                "total_allocated_rider_slots": total_allocated,
                "targeted_grid_count_sum": targeted_grid_count,
                "waiting_order_coverage_pct": covered_waiting / total_waiting * 100,
                "expected_late_risk_coverage_pct": covered_expected / total_expected * 100,
                "actual_late_order_coverage_pct_hindsight": covered_actual_late
                / total_actual_late
                * 100,
                "allocation_weighted_expected_late_rate_pct": weighted_expected_rate_numerator
                / total_allocated
                * 100,
                "allocation_weighted_actual_late_rate_pct_hindsight": weighted_actual_rate_numerator
                / total_allocated
                * 100,
                "mean_max_single_grid_allocation_share_pct": np.mean(max_concentration) * 100,
            }
        )

    scenario_summary = pd.DataFrame(summary_rows).sort_values(
        ["budget_rider_slots_per_epoch", "policy"]
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_grid.to_parquet(
        OUTPUT_DIR / "snapshot_grid_metrics.parquet", index=False, compression="zstd"
    )
    allocations.to_parquet(
        OUTPUT_DIR / "rider_slot_allocations.parquet", index=False, compression="zstd"
    )
    scenario_summary.to_csv(
        OUTPUT_DIR / "scenario_summary.csv", index=False, encoding="utf-8-sig"
    )
    snapshot_policy_metrics.to_csv(
        OUTPUT_DIR / "snapshot_policy_metrics.csv", index=False, encoding="utf-8-sig"
    )

    snapshot_profile = (
        snapshot_grid.groupby(["dt", "dispatch_time", "dispatch_time_local"], observed=True)
        .agg(
            waiting_orders=("waiting_orders", "sum"),
            model_scored_orders=("model_scored_orders", "sum"),
            demand_grids=("grid_x", "size"),
            actual_late_orders=("actual_late_orders", "sum"),
            mean_candidates_within_2km=("candidate_riders_within_2km", "mean"),
            mean_idle_candidates_within_2km=("idle_candidate_riders_within_2km", "mean"),
        )
        .reset_index()
    )
    snapshot_profile.to_csv(
        OUTPUT_DIR / "snapshot_profile.csv", index=False, encoding="utf-8-sig"
    )

    profile = {
        "evaluation_dates": [20221023, 20221024],
        "epochs": int(snapshot_grid[["dt", "dispatch_time"]].drop_duplicates().shape[0]),
        "waiting_orders": int(snapshot_grid["waiting_orders"].sum()),
        "model_scored_orders": int(snapshot_grid["model_scored_orders"].sum()),
        "model_scored_order_share_pct": round(
            float(
                snapshot_grid["model_scored_orders"].sum()
                / snapshot_grid["waiting_orders"].sum()
                * 100
            ),
            4,
        ),
        "actual_late_orders": int(snapshot_grid["actual_late_orders"].sum()),
        "grid_size_meters": [GRID_X_METERS, GRID_Y_METERS],
        "supply_radius_meters": SUPPLY_RADIUS_METERS,
        "smoothing_strength_orders": SMOOTHING_STRENGTH,
        "budgets_per_epoch": BUDGETS,
        "scenario_summary": json.loads(scenario_summary.to_json(orient="records")),
    }
    (OUTPUT_DIR / "capacity_profile.json").write_text(
        json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(snapshot_profile.to_string(index=False))
    print("\n", scenario_summary.to_string(index=False))


if __name__ == "__main__":
    main()
