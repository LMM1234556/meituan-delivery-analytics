from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "01_order_mart_and_sample_rules.ipynb"
NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    md(
        """
# 01 订单级分析宽表与样本规则

本 Notebook 验收从原始运单表加工出的订单级宽表，并冻结后续诊断和运力策略使用的样本。

业务问题不是“删掉多少异常”，而是：不同分析目的应该使用哪个分母，以及筛选是否会系统性改变结论。
"""
    ),
    md(
        """
## 管理摘要

- 订单宽表共 **568,546 行**，`order_id` 完全唯一；其中 **568,545 单**有完整送达结果。
- 高峰运力策略主样本定义为：**有效完成 + 即时单 + 已知商圈**，共 **436,062 单**。
- 全部有效完成订单准时率为 **84.99%**，策略样本为 **86.37%**。筛选会提高准时率，因此策略样本不能替代全城经营大盘。
- 预订单准时率约 **79.32%**，即时单约 **85.22%**；未知商圈订单准时率约 **80.69%**，低于已知商圈约 **86.09%**。该结果为描述性差异，不能直接解释为因果关系。
- 一条已完成订单缺失派单时间，仍纳入准时率，但不进入派单阶段时长分析。
"""
    ),
    code(
        """
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

pd.set_option("display.max_columns", 100)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
plt.style.use("seaborn-v0_8-whitegrid")

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "data").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

MART_PATH = PROJECT_ROOT / "data" / "processed" / "order_fulfillment_mart.parquet"
assert MART_PATH.exists(), MART_PATH
orders = pd.read_parquet(MART_PATH)

print(f"rows={len(orders):,}, columns={orders.shape[1]}, unique_orders={orders['order_id'].nunique():,}")
orders.head(3)
"""
    ),
    md(
        """
## 1. 主键、时间链路和覆盖检查

这里用硬性断言把主表契约写进分析流程。后续任何加工破坏唯一性或覆盖数，Notebook 都应执行失败。
"""
    ),
    code(
        """
assert orders["order_id"].is_unique
assert len(orders) == 568_546
assert int(orders["is_valid_completed"].sum()) == 568_545
assert int(orders["wave_id"].notna().sum()) == 568_545
assert int(orders["is_dispatch_timeline_valid"].sum()) == 568_545

quality_contract = pd.Series({
    "order_rows": len(orders),
    "unique_orders": orders["order_id"].nunique(),
    "valid_completed": int(orders["is_valid_completed"].sum()),
    "wave_matches": int(orders["wave_id"].notna().sum()),
    "valid_dispatch_timeline": int(orders["is_dispatch_timeline_valid"].sum()),
    "duplicate_orders": int(orders["order_id"].duplicated().sum()),
}, name="value")
quality_contract.to_frame()
"""
    ),
    md(
        """
## 2. 样本漏斗

三个筛选各自解决不同问题：

- 有效完成：结果指标有可靠分母；
- 即时单：匹配高峰即时需求与运力配置场景；
- 已知商圈：策略能落到具体运营单元。
"""
    ),
    code(
        """
sample_funnel = pd.DataFrame([
    {"stage": "Accepted order", "orders": len(orders)},
    {"stage": "Valid completed", "orders": int(orders["is_valid_completed"].sum())},
    {"stage": "Valid + on-demand", "orders": int((orders["is_valid_completed"] & orders["is_on_demand"]).sum())},
    {"stage": "Strategy eligible", "orders": int(orders["is_strategy_eligible"].sum())},
])
sample_funnel["retained_vs_all_pct"] = sample_funnel["orders"].div(len(orders)).mul(100)
sample_funnel
"""
    ),
    code(
        """
ax = sample_funnel.plot(
    x="stage", y="orders", kind="bar", legend=False, figsize=(9, 4), color="#2F6B9A"
)
ax.set_title("Order sample funnel")
ax.set_xlabel("")
ax.set_ylabel("Orders")
ax.tick_params(axis="x", rotation=20)
for container in ax.containers:
    ax.bar_label(container, fmt="{:,.0f}")
plt.tight_layout()
plt.show()
"""
    ),
    md(
        """
## 3. 筛选偏差检查

如果被排除样本的准时率不同，只报策略样本会让全城水平看起来更好或更差。因此总体经营指标和可执行策略样本必须并列报告。
"""
    ),
    code(
        """
valid = orders.loc[orders["is_valid_completed"]].copy()

segment_comparison = pd.concat([
    valid.groupby("is_prebook", observed=True).agg(
        orders=("order_id", "size"), on_time_rate=("is_on_time", "mean")
    ).rename(index={0: "on_demand", 1: "prebook"}),
    valid.groupby("is_known_da", observed=True).agg(
        orders=("order_id", "size"), on_time_rate=("is_on_time", "mean")
    ).rename(index={False: "unknown_da", True: "known_da"}),
])
segment_comparison["on_time_rate_pct"] = segment_comparison["on_time_rate"].mul(100)
segment_comparison
"""
    ),
    code(
        """
kpi_scope = pd.DataFrame([
    {
        "scope": "all_valid_completed",
        "orders": len(valid),
        "on_time_rate_pct": valid["is_on_time"].mean() * 100,
    },
    {
        "scope": "strategy_eligible",
        "orders": int(orders["is_strategy_eligible"].sum()),
        "on_time_rate_pct": orders.loc[orders["is_strategy_eligible"], "is_on_time"].mean() * 100,
    },
])
kpi_scope
"""
    ),
    md(
        """
**解释边界：** 预订单和未知商圈的准时率较低是已观察到的相关性，不足以证明“预订”或“区域未知”导致超时。可能还混入时段、距离、商家、承诺时长和数据采集机制差异。
"""
    ),
    md(
        """
## 4. 订单级过程指标分布

所有时长均保留原值。表格报告 P50/P90/P99/最大值；后续画图可以截尾，但不能悄悄删除长尾。
"""
    ),
    code(
        """
stage_columns = [
    "order_to_push_minutes",
    "first_dispatch_wait_minutes",
    "accepted_dispatch_wait_minutes",
    "accepted_dispatch_to_grab_minutes",
    "accept_to_pickup_minutes",
    "pickup_to_delivery_minutes",
    "total_fulfillment_minutes",
    "promise_minutes",
    "late_minutes",
]

strategy = orders.loc[orders["is_strategy_eligible"]].copy()
stage_summary = strategy[stage_columns].quantile([0.5, 0.9, 0.99, 0.999, 1.0]).T
stage_summary.columns = ["p50", "p90", "p99", "p999", "max"]
stage_summary
"""
    ),
    code(
        """
distance_summary = strategy[
    ["merchant_customer_distance_km", "courier_merchant_distance_at_accept_km"]
].quantile([0.5, 0.9, 0.99, 0.999, 1.0]).T
distance_summary.columns = ["p50", "p90", "p99", "p999", "max"]
distance_summary
"""
    ),
    md(
        """
## 5. 超时严重程度

严重程度阈值是项目分析分层，不是美团内部 SLA。准时率之外必须同时看超时订单规模和长尾。
"""
    ),
    code(
        """
severity = (
    valid["late_severity"]
    .value_counts(sort=False)
    .rename_axis("late_severity")
    .to_frame("orders")
)
severity["share_pct"] = severity["orders"].div(len(valid)).mul(100)
severity
"""
    ),
    code(
        """
ax = severity["orders"].plot(kind="bar", figsize=(9, 4), color="#D05A3A")
ax.set_title("Late severity distribution")
ax.set_xlabel("")
ax.set_ylabel("Orders")
ax.tick_params(axis="x", rotation=15)
for container in ax.containers:
    ax.bar_label(container, fmt="{:,.0f}")
plt.tight_layout()
plt.show()
"""
    ),
    md(
        """
## 6. 宽表使用契约

| 场景 | 样本 | 注意事项 |
|---|---|---|
| 全城准时率 | `is_valid_completed` | 包含预订单和未知商圈 |
| 商圈时段诊断 | `is_strategy_eligible` | 结果不能外推为全城大盘 |
| 派单阶段分析 | 再加 `is_dispatch_timeline_valid` | 缺失派单时点的订单不参与该阶段 |
| 预订单专项 | `is_valid_completed & ~is_on_demand` | 与即时单分开分析 |
| 模型训练 | 按业务日期切分策略样本 | 禁止使用随机切分替代跨期留出验证 |

下一步在策略样本上建立“全城 → 日期 → 小时 → 商圈 × 30 分钟”的指标树，并判断哪些延误阶段与运力更相关。
"""
    ),
    md(
        """
## Sources

- `data/processed/order_fulfillment_mart.parquet`
- `docs/05_sample_and_table_rules.md`
- `scripts/build_order_mart.py`
"""
    ),
]

notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
)
nbf.write(notebook, NOTEBOOK_PATH)
print(NOTEBOOK_PATH)
