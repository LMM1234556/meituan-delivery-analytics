from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "outputs" / "metrics"
MODEL = ROOT / "outputs" / "model"
CAPACITY = ROOT / "outputs" / "capacity"
FIGURES = ROOT / "assets" / "figures"

BLUE = "#2F5D8A"
LIGHT_BLUE = "#8DB3D3"
GOLD = "#C9952E"
GREY = "#9AA3AD"
LIGHT_GREY = "#E6E9ED"
TEXT = "#20262E"


def configure_style() -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in ["Microsoft YaHei", "DengXian", "SimHei"]:
        if candidate in available:
            plt.rcParams["font.sans-serif"] = [candidate]
            break
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#7E8791",
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "xtick.color": "#4D5660",
            "ytick.color": "#4D5660",
            "text.color": TEXT,
            "font.size": 10.5,
            "axes.titlesize": 12,
            "figure.titlesize": 16,
        }
    )


def clean_axes(ax: plt.Axes, axis: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=axis, color=LIGHT_GREY, linewidth=0.8)
    ax.set_axisbelow(True)


def add_source(fig: plt.Figure, text: str) -> None:
    fig.text(0.01, 0.012, text, ha="left", va="bottom", fontsize=8, color="#68717B")


def save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_peak_figure(profile: dict) -> None:
    share_values = [profile["peak_order_share_pct"], profile["peak_late_order_share_pct"]]
    rate_values = [profile["peak_late_rate_pct"], profile["off_peak_late_rate_pct"]]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"wspace": 0.32})
    fig.suptitle("用餐高峰承载 71.70% 的订单，却贡献 77.33% 的超时订单", y=0.98)

    bars = axes[0].bar(["订单量占比", "超时订单占比"], share_values, color=[LIGHT_BLUE, BLUE], width=0.58)
    axes[0].set_title("高峰在全量样本中的占比")
    axes[0].set_ylabel("占比（%）")
    axes[0].set_ylim(0, 100)
    axes[0].bar_label(bars, labels=[f"{v:.2f}%" for v in share_values], padding=4, fontsize=11)
    clean_axes(axes[0])

    bars = axes[1].bar(["用餐高峰", "非高峰"], rate_values, color=[BLUE, GREY], width=0.58)
    axes[1].set_title("高峰与非高峰超时率")
    axes[1].set_ylabel("超时率（%）")
    axes[1].set_ylim(0, max(rate_values) * 1.35)
    axes[1].bar_label(bars, labels=[f"{v:.2f}%" for v in rate_values], padding=4, fontsize=11)
    clean_axes(axes[1])

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.14, top=0.84, wspace=0.32)
    add_source(fig, "口径：策略样本；高峰为 [10:30, 14:00)、[17:00, 21:00)。生成逻辑：scripts/build_metric_marts.py")
    save(fig, "01_peak_concentration.png")


def build_district_figure(district: pd.DataFrame) -> None:
    district = district.copy()
    district["late_rate_pct"] = district["late_rate"] * 100
    district["orders_thousand"] = district["orders"] / 1_000
    district["is_top5"] = district["late_orders"].rank(method="first", ascending=False).le(5)
    top5_order_share = district.loc[district["is_top5"], "orders"].sum() / district["orders"].sum() * 100
    top5_late_share = (
        district.loc[district["is_top5"], "late_orders"].sum()
        / district["late_orders"].sum()
        * 100
    )
    sizes = 40 + district["late_orders"] / district["late_orders"].max() * 430

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    colors = np.where(district["is_top5"], BLUE, LIGHT_BLUE)
    ax.scatter(
        district["orders_thousand"],
        district["late_rate_pct"],
        s=sizes,
        c=colors,
        alpha=0.82,
        edgecolor="white",
        linewidth=0.9,
    )
    ax.axvline(district["orders_thousand"].median(), color=GREY, linestyle="--", linewidth=1)
    ax.axhline(district["late_rate_pct"].median(), color=GREY, linestyle="--", linewidth=1)
    for _, row in district.loc[district["is_top5"]].iterrows():
        ax.annotate(
            f"da{int(row['da_id'])}",
            (row["orders_thousand"], row["late_rate_pct"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_title(
        f"前五个匿名商圈以 {top5_order_share:.2f}% 的订单贡献 {top5_late_share:.2f}% 的超时订单",
        pad=14,
        fontsize=15,
    )
    ax.set_xlabel("订单量（千单）")
    ax.set_ylabel("超时率（%）")
    ax.text(
        0.99,
        0.02,
        "气泡大小表示超时订单量；虚线为商圈中位数",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#68717B",
    )
    clean_axes(ax)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    add_source(fig, "样本：有效完成、即时配送、匿名商圈已知订单。生成逻辑：scripts/build_metric_marts.py")
    save(fig, "02_district_priority.png")


def build_stage_figure(stage: pd.DataFrame) -> None:
    labels = {
        "order_to_push_minutes": "下单至进入配送",
        "order_push_to_accept_minutes": "进入配送至接单",
        "accept_to_pickup_minutes": "接单至取餐",
        "pickup_to_delivery_minutes": "取餐至送达",
    }
    stage = stage.copy()
    stage["label"] = stage["stage"].map(labels)
    stage["gap_share_pct"] = stage["gap_share"] * 100

    fig, ax = plt.subplots(figsize=(10, 5.3))
    bars = ax.barh(stage["label"], stage["mean_gap_minutes"], color=[LIGHT_BLUE] * 3 + [BLUE])
    ax.invert_yaxis()
    ax.set_title("超时单平均多用 17.32 分钟，57.11% 来自取餐至送达阶段", pad=14, fontsize=16)
    ax.set_xlabel("超时单与准时单的平均时长差（分钟）")
    ax.bar_label(
        bars,
        labels=[f"{g:.2f} 分钟（{s:.2f}%）" for g, s in zip(stage["mean_gap_minutes"], stage["gap_share_pct"])],
        padding=5,
        fontsize=10,
    )
    ax.set_xlim(0, stage["mean_gap_minutes"].max() * 1.42)
    clean_axes(ax, axis="x")
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    add_source(fig, "说明：均值差分解属于描述性比较，不表示各阶段对超时的因果贡献。生成逻辑：scripts/build_metric_marts.py")
    save(fig, "03_stage_gap.png")


def build_model_figure(model: pd.DataFrame) -> None:
    model = model.loc[model["split"].eq("test_20221023_20221024")].copy()
    order = ["distance_only_ranking", "push_time_model", "acceptance_time_model"]
    names = {"distance_only_ranking": "距离排序基线", "push_time_model": "订单推送时模型", "acceptance_time_model": "骑手接单时模型"}
    model["model"] = pd.Categorical(model["model"], categories=order, ordered=True)
    model = model.sort_values("model")
    colors = [GREY, LIGHT_BLUE, BLUE]

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.2), gridspec_kw={"wspace": 0.38})
    fig.suptitle("骑手接单时模型在跨期测试中的风险排序能力最强", y=0.98)
    labels = [names[str(v)] for v in model["model"]]
    for ax, column, title, fmt in [
        (axes[0], "pr_auc", "PR-AUC", "{:.3f}"),
        (axes[1], "lift_at_top_10pct", "Lift@Top10%", "{:.2f}"),
    ]:
        bars = ax.barh(labels, model[column], color=colors, height=0.58)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.bar_label(bars, labels=[fmt.format(v) for v in model[column]], padding=5, fontsize=10.5)
        ax.set_xlim(0, model[column].max() * 1.25)
        clean_axes(ax, axis="x")
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.14, top=0.84, wspace=0.38)
    add_source(fig, "跨期测试：2022-10-23 至 2022-10-24，共 111,835 笔订单。生成逻辑：scripts/train_late_risk_model.py")
    save(fig, "04_model_performance.png")


def build_capacity_figure(summary: pd.DataFrame, snapshot: pd.DataFrame) -> None:
    names = {"risk_adjusted": "风险加权策略", "volume_only": "订单量基线"}
    colors = {"risk_adjusted": BLUE, "volume_only": GREY}
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.7), gridspec_kw={"height_ratios": [1, 1.18], "hspace": 0.42})
    fig.suptitle("风险加权策略在汇总结果中覆盖更多后验超时订单，但证据仅来自六个快照", y=0.985)

    for policy, group in summary.groupby("policy", sort=False):
        group = group.sort_values("budget_rider_slots_per_epoch")
        axes[0].plot(
            group["budget_rider_slots_per_epoch"],
            group["actual_late_order_coverage_pct_hindsight"],
            marker="o",
            linewidth=2.2,
            markersize=6,
            color=colors[policy],
            label=names[policy],
        )
        for x, y in zip(group["budget_rider_slots_per_epoch"], group["actual_late_order_coverage_pct_hindsight"]):
            axes[0].annotate(f"{y:.2f}%", (x, y), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=9)
    axes[0].set_title("不同预算下的汇总后验超时订单覆盖率")
    axes[0].set_xlabel("每个快照的假设增量运力槽位")
    axes[0].set_ylabel("后验超时订单覆盖率（%）")
    axes[0].set_xticks(sorted(summary["budget_rider_slots_per_epoch"].unique()))
    axes[0].legend(frameon=False, ncol=2, loc="upper left")
    clean_axes(axes[0])

    detail = snapshot.loc[snapshot["budget_rider_slots"].eq(30)].copy()
    detail["epoch"] = pd.to_datetime(detail["dispatch_time_local"]).dt.strftime("%m-%d\n%H:%M:%S")
    pivot = detail.pivot(index="epoch", columns="policy", values="actual_late_order_coverage_pct_hindsight")
    x = np.arange(len(pivot))
    width = 0.34
    risk_bars = axes[1].bar(x - width / 2, pivot["risk_adjusted"], width, color=BLUE, label=names["risk_adjusted"])
    volume_bars = axes[1].bar(x + width / 2, pivot["volume_only"], width, color=GREY, label=names["volume_only"])
    axes[1].bar_label(risk_bars, labels=[f"{v:.1f}%" for v in pivot["risk_adjusted"]], padding=3, fontsize=8.5)
    axes[1].bar_label(volume_bars, labels=[f"{v:.1f}%" for v in pivot["volume_only"]], padding=3, fontsize=8.5)
    axes[1].set_title("30 个槽位预算下的逐快照比较：风险策略 4 胜、1 平、1 负")
    axes[1].set_ylabel("后验超时订单覆盖率（%）")
    axes[1].set_xticks(x, pivot.index)
    axes[1].legend(frameon=False, ncol=2, loc="upper right")
    axes[1].set_ylim(0, max(pivot.max()) * 1.25)
    clean_axes(axes[1])

    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.10, top=0.91, hspace=0.48)
    add_source(fig, "回放范围：2022-10-23 至 2022-10-24 的六个午高峰派单快照。生成逻辑：scripts/build_capacity_scenario.py；后验覆盖率不代表超时减少量。")
    save(fig, "05_capacity_strategy.png")


def main() -> None:
    configure_style()
    profile = json.loads((METRICS / "metric_diagnostic_profile.json").read_text(encoding="utf-8"))
    district = pd.read_csv(METRICS / "district_kpi.csv")
    stage = pd.read_csv(METRICS / "late_vs_on_time_stage_gap.csv")
    model = pd.read_csv(MODEL / "model_metrics.csv")
    summary = pd.read_csv(CAPACITY / "scenario_summary.csv")
    snapshot = pd.read_csv(CAPACITY / "snapshot_policy_metrics.csv")

    build_peak_figure(profile)
    build_district_figure(district)
    build_stage_figure(stage)
    build_model_figure(model)
    build_capacity_figure(summary, snapshot)
    print(f"Wrote 5 report figures to {FIGURES}")


if __name__ == "__main__":
    main()
