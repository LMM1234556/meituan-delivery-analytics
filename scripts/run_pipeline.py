from __future__ import annotations

import argparse
import subprocess
import sys
from zipfile import ZipFile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "downloads"
PYTHON = Path(sys.executable)

REQUIRED_RAW_FILES = [
    "all_waybill_info_meituan_0322.csv.zip",
    "courier_wave_info_meituan.csv",
    "dispatch_rider_meituan.csv",
    "dispatch_waybill_meituan.csv",
]

CORE_STEPS = [
    "profile_raw_data.py",
    "build_order_mart.py",
    "build_metric_marts.py",
    "train_late_risk_model.py",
    "build_capacity_scenario.py",
    "run_sql_checks.py",
    "build_report_snapshot.py",
    "build_report_figures.py",
]

NOTEBOOK_BUILDERS = [
    "build_00_data_inventory_notebook.py",
    "build_01_order_mart_notebook.py",
    "build_02_fulfillment_diagnostic_notebook.py",
    "build_03_late_risk_model_notebook.py",
    "build_04_capacity_scenario_notebook.py",
]

NOTEBOOKS = [
    "00_data_inventory.ipynb",
    "01_order_mart_and_sample_rules.ipynb",
    "02_fulfillment_metric_diagnostic.ipynb",
    "03_late_risk_model.ipynb",
    "04_peak_capacity_scenario.ipynb",
]


def check_inputs() -> None:
    missing = [name for name in REQUIRED_RAW_FILES if not (RAW_DIR / name).exists()]
    if missing:
        formatted = "\n".join(f"  - {RAW_DIR / name}" for name in missing)
        raise FileNotFoundError(
            "缺少以下原始数据文件，请先按照 data/raw/README.md 下载：\n" + formatted
        )


def ensure_waybill_extracted() -> None:
    target_dir = RAW_DIR / "waybill_extracted"
    target_path = target_dir / "all_waybill_info_meituan_0322.csv"
    if target_path.exists():
        return

    archive_path = RAW_DIR / "all_waybill_info_meituan_0322.csv.zip"
    target_dir.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive_path) as archive:
        member = "all_waybill_info_meituan_0322.csv"
        if member not in archive.namelist():
            raise FileNotFoundError(f"压缩包中缺少预期文件：{member}")
        archive.extract(member, path=target_dir)
    print(f"[PREP] 已解压运单表：{target_path}", flush=True)


def run(command: list[str]) -> None:
    print("\n[RUN]", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="复现履约诊断、风险模型和运力策略回放。")
    parser.add_argument(
        "--skip-notebooks",
        action="store_true",
        help="只生成数据、指标、模型和策略结果，不重新执行 Notebook。",
    )
    args = parser.parse_args()

    check_inputs()
    ensure_waybill_extracted()

    for script_name in CORE_STEPS:
        run([str(PYTHON), str(ROOT / "scripts" / script_name)])

    if not args.skip_notebooks:
        for script_name in NOTEBOOK_BUILDERS:
            run([str(PYTHON), str(ROOT / "scripts" / script_name)])

        preview_dir = ROOT / "outputs" / "notebook_previews"
        preview_dir.mkdir(parents=True, exist_ok=True)
        for notebook_name in NOTEBOOKS:
            notebook_path = ROOT / "notebooks" / notebook_name
            run(
                [
                    str(PYTHON),
                    "-m",
                    "jupyter",
                    "nbconvert",
                    "--execute",
                    "--to",
                    "notebook",
                    "--inplace",
                    str(notebook_path),
                    "--ExecutePreprocessor.timeout=600",
                ]
            )
            run(
                [
                    str(PYTHON),
                    "-m",
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "html",
                    str(notebook_path),
                    "--output-dir",
                    str(preview_dir),
                    "--output",
                    f"{notebook_path.stem}.html",
                ]
            )

    print("\n[DONE] 项目流水线执行完成。", flush=True)


if __name__ == "__main__":
    main()
