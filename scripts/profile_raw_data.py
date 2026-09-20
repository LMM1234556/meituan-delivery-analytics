"""Profile the raw Meituan delivery files before any business analysis.

This script produces an inspectable JSON profile and a compact Markdown inventory.
It does not modify the raw source files.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "downloads"
WAYBILL_PATH = RAW_DIR / "waybill_extracted" / "all_waybill_info_meituan_0322.csv"
INPUT_PATHS = {
    "waybill": WAYBILL_PATH,
    "courier_wave": RAW_DIR / "courier_wave_info_meituan.csv",
    "dispatch_rider": RAW_DIR / "dispatch_rider_meituan.csv",
    "dispatch_waybill": RAW_DIR / "dispatch_waybill_meituan.csv",
}
OUTPUT_JSON = PROJECT_ROOT / "outputs" / "raw_data_profile.json"
OUTPUT_MD = PROJECT_ROOT / "docs" / "04_raw_data_inventory.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    unnamed = [column for column in frame.columns if str(column).startswith("Unnamed:")]
    if len(unnamed) == 1:
        frame = frame.rename(columns={unnamed[0]: "source_row_id"})
    return frame


def serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def table_profile(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    null_rates = (frame.isna().mean() * 100).round(4).to_dict()
    distinct_counts = frame.nunique(dropna=True).to_dict()
    zero_rates: dict[str, float] = {}
    for column in frame.select_dtypes(include="number").columns:
        zero_rates[column] = round(float(frame[column].eq(0).mean() * 100), 4)
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "rows": len(frame),
        "columns": len(frame.columns),
        "column_names": frame.columns.tolist(),
        "dtypes": frame.dtypes.astype(str).to_dict(),
        "null_rate_pct": null_rates,
        "zero_rate_pct_numeric": zero_rates,
        "distinct_counts": distinct_counts,
    }


def nonzero_timestamp_range(frame: pd.DataFrame, column: str) -> dict[str, str | int | None]:
    values = pd.to_numeric(frame[column], errors="coerce")
    values = values[values.gt(0)]
    if values.empty:
        return {"count": 0, "min_shanghai": None, "max_shanghai": None}
    converted = pd.to_datetime(values, unit="s", utc=True).dt.tz_convert("Asia/Shanghai")
    return {
        "count": int(len(values)),
        "min_shanghai": converted.min().isoformat(),
        "max_shanghai": converted.max().isoformat(),
    }


def parse_order_ids(raw_value: Any) -> list[int]:
    if pd.isna(raw_value):
        return []
    parsed = ast.literal_eval(str(raw_value))
    if not isinstance(parsed, list):
        return []
    return [int(value) for value in parsed]


def build_profile() -> dict[str, Any]:
    frames = {name: load_csv(path) for name, path in INPUT_PATHS.items()}
    waybill = frames["waybill"]
    courier_wave = frames["courier_wave"]
    dispatch_rider = frames["dispatch_rider"]
    dispatch_waybill = frames["dispatch_waybill"]

    profile: dict[str, Any] = {
        "tables": {
            name: table_profile(frame, INPUT_PATHS[name])
            for name, frame in frames.items()
        }
    }

    waybill_key_duplicates = int(waybill.duplicated(subset=["waybill_id"], keep=False).sum())
    order_waybill_counts = waybill.groupby("order_id")["waybill_id"].nunique()
    successful = waybill.loc[waybill["is_courier_grabbed"].eq(1)].copy()
    success_counts_per_order = successful.groupby("order_id")["waybill_id"].nunique()
    completed = successful.loc[
        successful["arrive_time"].gt(0) & successful["estimate_arrived_time"].gt(0)
    ].copy()
    completed["is_on_time"] = completed["arrive_time"].le(completed["estimate_arrived_time"])

    timestamp_columns = [
        "platform_order_time",
        "order_push_time",
        "dispatch_time",
        "grab_time",
        "fetch_time",
        "arrive_time",
        "estimate_arrived_time",
        "estimate_meal_prepare_time",
    ]
    chronology_pairs = [
        ("platform_order_time", "order_push_time"),
        ("order_push_time", "dispatch_time"),
        ("dispatch_time", "grab_time"),
        ("grab_time", "fetch_time"),
        ("fetch_time", "arrive_time"),
    ]
    chronology: dict[str, Any] = {}
    for start, end in chronology_pairs:
        eligible = successful[start].gt(0) & successful[end].gt(0)
        violations = eligible & successful[end].lt(successful[start])
        chronology[f"{start}_le_{end}"] = {
            "eligible_rows": int(eligible.sum()),
            "violation_rows": int(violations.sum()),
            "violation_rate_pct": round(float(violations.sum() / max(eligible.sum(), 1) * 100), 6),
        }

    profile["waybill_grain"] = {
        "rows": int(len(waybill)),
        "unique_waybill_ids": int(waybill["waybill_id"].nunique()),
        "waybill_id_duplicate_rows": waybill_key_duplicates,
        "unique_order_ids": int(waybill["order_id"].nunique()),
        "orders_with_multiple_waybills": int(order_waybill_counts.gt(1).sum()),
        "max_waybills_per_order": int(order_waybill_counts.max()),
        "grabbed_rows": int(len(successful)),
        "grabbed_unique_orders": int(successful["order_id"].nunique()),
        "orders_with_multiple_grabbed_waybills": int(success_counts_per_order.gt(1).sum()),
        "completed_rows_with_eta": int(len(completed)),
        "preliminary_on_time_rate_pct": round(float(completed["is_on_time"].mean() * 100), 4),
        "date_min": int(waybill["dt"].min()),
        "date_max": int(waybill["dt"].max()),
        "daily_rows": waybill.groupby("dt").size().astype(int).to_dict(),
        "distinct_business_districts": int(waybill["da_id"].nunique()),
        "distinct_couriers_nonzero": int(waybill.loc[waybill["courier_id"].ne(0), "courier_id"].nunique()),
        "distinct_merchants": int(waybill["poi_id"].nunique()),
        "timestamp_ranges": {
            column: nonzero_timestamp_range(waybill, column)
            for column in timestamp_columns
        },
        "chronology_checks_on_grabbed_rows": chronology,
    }

    profile["candidate_keys"] = {
        "courier_wave_dt_courier_wave_duplicate_rows": int(
            courier_wave.duplicated(subset=["dt", "courier_id", "wave_id"], keep=False).sum()
        ),
        "dispatch_rider_source_row_id_duplicate_rows": int(
            dispatch_rider.duplicated(subset=["source_row_id"], keep=False).sum()
        ),
        "dispatch_waybill_source_row_id_duplicate_rows": int(
            dispatch_waybill.duplicated(subset=["source_row_id"], keep=False).sum()
        ),
        "dispatch_waybill_dispatch_time_order_id_duplicate_rows": int(
            dispatch_waybill.duplicated(subset=["dispatch_time", "order_id"], keep=False).sum()
        ),
    }

    known_orders = set(waybill["order_id"].astype(int))
    successful_orders = set(successful["order_id"].astype(int))
    dispatch_orders = set(dispatch_waybill["order_id"].astype(int))
    dispatch_key = ["dt", "dispatch_time"]
    rider_counts = dispatch_rider.groupby(dispatch_key).size()
    order_counts = dispatch_waybill.groupby(dispatch_key).size()
    common_dispatch_epochs = rider_counts.index.intersection(order_counts.index)
    joined_rows_on_epoch = int(
        (rider_counts.loc[common_dispatch_epochs] * order_counts.loc[common_dispatch_epochs]).sum()
    )
    profile["dispatch_integrity"] = {
        "dispatch_waybill_rows": int(len(dispatch_waybill)),
        "dispatch_waybill_unique_orders": int(dispatch_waybill["order_id"].nunique()),
        "dispatch_orders_found_in_waybill_pct": round(
            float(len(dispatch_orders & known_orders) / max(len(dispatch_orders), 1) * 100), 4
        ),
        "dispatch_rider_rows": int(len(dispatch_rider)),
        "dispatch_rider_unique_epochs": int(dispatch_rider[dispatch_key].drop_duplicates().shape[0]),
        "dispatch_waybill_unique_epochs": int(
            dispatch_waybill[dispatch_key].drop_duplicates().shape[0]
        ),
        "common_dispatch_epochs": int(len(common_dispatch_epochs)),
        "estimated_rows_if_joined_on_dispatch_epoch": joined_rows_on_epoch,
        "dispatch_rider_rows_per_time_p50": float(rider_counts.quantile(0.5)),
        "dispatch_rider_rows_per_time_p90": float(rider_counts.quantile(0.9)),
        "dispatch_waybill_rows_per_time_max": int(order_counts.max()),
    }

    wave_lists = courier_wave["order_ids"].map(parse_order_ids)
    flattened_wave_orders = [order for order_list in wave_lists for order in order_list]
    unique_wave_orders = set(flattened_wave_orders)
    profile["courier_wave_integrity"] = {
        "wave_rows": int(len(courier_wave)),
        "unique_couriers": int(courier_wave["courier_id"].nunique()),
        "total_order_references": int(len(flattened_wave_orders)),
        "unique_order_references": int(len(unique_wave_orders)),
        "duplicate_order_references": int(len(flattened_wave_orders) - len(unique_wave_orders)),
        "order_references_found_in_waybill_pct": round(
            float(len(unique_wave_orders & successful_orders) / max(len(unique_wave_orders), 1) * 100),
            4,
        ),
        "successful_orders_found_in_wave_pct": round(
            float(len(successful_orders & unique_wave_orders) / max(len(successful_orders), 1) * 100),
            4,
        ),
        "successful_orders_missing_from_wave": int(len(successful_orders - unique_wave_orders)),
        "orders_per_wave_mean": round(float(wave_lists.map(len).mean()), 4),
        "orders_per_wave_p90": float(wave_lists.map(len).quantile(0.9)),
        "orders_per_wave_max": int(wave_lists.map(len).max()),
        "wave_start_after_end_rows": int(
            courier_wave["wave_start_time"].gt(courier_wave["wave_end_time"]).sum()
        ),
    }

    return serializable(profile)


def build_markdown(profile: dict[str, Any]) -> str:
    tables = profile["tables"]
    waybill = profile["waybill_grain"]
    dispatch = profile["dispatch_integrity"]
    waves = profile["courier_wave_integrity"]
    chronology = waybill["chronology_checks_on_grabbed_rows"]
    chronology_violation_rows = sum(item["violation_rows"] for item in chronology.values())

    lines = [
        "# 04 原始数据盘点",
        "",
        "> 本文档由 `scripts/profile_raw_data.py` 从只读原始文件生成。当前结论属于接入检查，不是最终业务结论。",
        "",
        "## 1. 数据集与粒度摘要",
        "",
        "| 表 | 行数 | 列数 | 初步粒度 |",
        "|---|---:|---:|---|",
        f"| waybill | {tables['waybill']['rows']:,} | {tables['waybill']['columns']} | 一行一条运单记录；订单可能对应多条运单 |",
        f"| courier_wave | {tables['courier_wave']['rows']:,} | {tables['courier_wave']['columns']} | 骑手在某日的一次配送波次 |",
        f"| dispatch_rider | {tables['dispatch_rider']['rows']:,} | {tables['dispatch_rider']['columns']} | 某派单时刻的候选骑手记录 |",
        f"| dispatch_waybill | {tables['dispatch_waybill']['rows']:,} | {tables['dispatch_waybill']['columns']} | 某派单时刻关联的订单记录 |",
        "",
        "## 2. 关键检查结果",
        "",
        f"- 运单表覆盖 `{waybill['date_min']}` 至 `{waybill['date_max']}`，共 {len(waybill['daily_rows'])} 个日期分区。",
        f"- 运单表包含 {waybill['rows']:,} 行、{waybill['unique_order_ids']:,} 个不同 `order_id`、{waybill['unique_waybill_ids']:,} 个不同 `waybill_id`；业务含义由字段说明及后续关系检查共同确认。",
        f"- {waybill['orders_with_multiple_waybills']:,} 个订单对应多条运单，因此不能直接按运单行数计算订单量。",
        f"- 抢单成功记录 {waybill['grabbed_rows']:,} 行，对应 {waybill['grabbed_unique_orders']:,} 个订单。",
        f"- 有实际送达和预计送达时间的成功记录为 {waybill['completed_rows_with_eta']:,} 行；初步准时率为 {waybill['preliminary_on_time_rate_pct']:.2f}%，该数值尚未经过最终有效订单口径冻结。",
        f"- 成功运单关键时间顺序检查累计发现 {chronology_violation_rows:,} 条阶段顺序异常命中；需要按规则去重后再决定是否剔除。",
        f"- 派单订单在运单表中的覆盖率为 {dispatch['dispatch_orders_found_in_waybill_pct']:.2f}%。",
        f"- 若按 `dt + dispatch_time` 直接逐行关联两张派单表，预计产生 {dispatch['estimated_rows_if_joined_on_dispatch_epoch']:,} 行，属于候选骑手与等待订单的多对多扩张。",
        f"- 骑手波次共引用 {waves['unique_order_references']:,} 个不同 `order_id`，重复引用 {waves['duplicate_order_references']:,} 条，且 {waves['order_references_found_in_waybill_pct']:.2f}% 可在成功运单中找到；反向覆盖率为 {waves['successful_orders_found_in_wave_pct']:.4f}%，缺少 {waves['successful_orders_missing_from_wave']:,} 笔。",
        "",
        "## 3. 当前数据风险",
        "",
        "### 高风险：订单与运单不是同一粒度",
        "",
        "一个订单可能对应多条运单。任何订单数、超时率或模型样本都必须先定义保留哪条成功运单，并检查同一订单是否存在多条成功记录。",
        "",
        "### 高风险：派单表可能发生多对多关联",
        "",
        "`dispatch_rider` 与 `dispatch_waybill` 共享 `dt + dispatch_time`，但两表并非逐行对应。直接关联会把同一快照内的候选骑手和等待订单做笛卡尔扩张；分析时仅在该快照粒度汇总供需，或进一步做空间邻域统计。",
        "",
        "### 中风险：0值是业务缺失哨兵",
        "",
        "未抢单运单的骑手、坐标和后续时间戳可能记录为0，而不是空值。仅检查SQL NULL会低估缺失率。",
        "",
        "### 中风险：八天数据限制",
        "",
        "数据适合订单流程和短周期商圈时段诊断，但不支持月度趋势、季节性和长期排班结论。",
        "",
        "## 4. 质量控制落地",
        "",
        "1. 订单宽表仅保留每个 `order_id` 的唯一成功运单，并聚合全部派单尝试；",
        "2. 波次表展开后按 `order_id` 做一对一校验，未使用内连接静默删除订单；",
        "3. 两张派单快照表仅按 `dt × dispatch_time` 汇总供需或做空间邻域统计，不做逐行订单—骑手连接；",
        "4. 仅对事件时间字段将 0 解释为业务缺失，类别与标记字段保留其原始业务含义；",
        "5. 准时率仅在成功接单且实际送达时间、预计送达时间和核心时间链路均有效的订单中计算。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    missing = [str(path) for path in INPUT_PATHS.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing raw files: {missing}")

    profile = build_profile()
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    OUTPUT_MD.write_text(build_markdown(profile), encoding="utf-8")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
