from __future__ import annotations

import compileall
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PATHS = [
    "README.md",
    "requirements.txt",
    "data/raw/README.md",
    "docs/01_project_charter.md",
    "docs/02_kpi_dictionary.md",
    "docs/03_analysis_roadmap.md",
    "docs/04_raw_data_inventory.md",
    "docs/05_sample_and_table_rules.md",
    "docs/06_model_design.md",
    "docs/07_capacity_scenario.md",
    "docs/08_final_business_report.md",
    "notebooks/00_data_inventory.ipynb",
    "notebooks/01_order_mart_and_sample_rules.ipynb",
    "notebooks/02_fulfillment_metric_diagnostic.ipynb",
    "notebooks/03_late_risk_model.ipynb",
    "notebooks/04_peak_capacity_scenario.ipynb",
    "scripts/run_pipeline.py",
    "scripts/build_report_figures.py",
    "assets/figures/01_peak_concentration.png",
    "assets/figures/02_district_priority.png",
    "assets/figures/03_stage_gap.png",
    "assets/figures/04_model_performance.png",
    "assets/figures/05_capacity_strategy.png",
    "sql/01_core_kpis.sql",
    "sql/02_district_time_metrics.sql",
    "sql/03_stage_gap_decomposition.sql",
]

FORBIDDEN_TRACKED_PATTERNS = [
    re.compile(r"^data/raw/downloads/"),
    re.compile(r"^data/processed/.+\.(parquet|csv)$", re.IGNORECASE),
    re.compile(r"^models/.+\.joblib$", re.IGNORECASE),
    re.compile(r"^outputs/(?!\.gitkeep$)"),
    re.compile(r"^tmp/"),
    re.compile(r"^report_app/"),
    re.compile(r"^\.venv/"),
]

PERSONAL_PATH_PATTERNS = [
    re.compile(r"[A-Za-z]:\\Users\\", re.IGNORECASE),
    re.compile(r"[A-Za-z]:\\[^\r\n\"']+\\meituan_delivery_analytics", re.IGNORECASE),
]

MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip().replace("\\", "/") for line in result.stdout.splitlines() if line.strip()]


def validate_required_paths(errors: list[str]) -> None:
    for relative_path in REQUIRED_PATHS:
        if not (ROOT / relative_path).exists():
            errors.append(f"缺少必需文件：{relative_path}")


def validate_tracked_scope(files: list[str], errors: list[str]) -> None:
    for relative_path in files:
        if any(pattern.search(relative_path) for pattern in FORBIDDEN_TRACKED_PATTERNS):
            errors.append(f"不应纳入版本控制：{relative_path}")


def validate_markdown_links(files: list[str], errors: list[str]) -> None:
    markdown_files = [ROOT / path for path in files if path.lower().endswith(".md")]
    for path in markdown_files:
        text = path.read_text(encoding="utf-8")
        for match in MARKDOWN_LINK.finditer(text):
            target = match.group(1).strip().split("#", 1)[0]
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            resolved = (path.parent / target).resolve()
            if not resolved.exists():
                errors.append(f"失效的本地链接：{path.relative_to(ROOT)} -> {target}")


def validate_notebooks(files: list[str], errors: list[str]) -> tuple[int, int]:
    notebook_count = 0
    output_count = 0
    for relative_path in files:
        if not relative_path.endswith(".ipynb"):
            continue
        notebook_count += 1
        notebook = json.loads((ROOT / relative_path).read_text(encoding="utf-8"))
        cells = notebook.get("cells", [])
        if not cells:
            errors.append(f"Notebook 没有单元格：{relative_path}")
            continue
        for cell in cells:
            for output in cell.get("outputs", []):
                output_count += 1
                if output.get("output_type") == "error":
                    errors.append(
                        f"Notebook 包含错误输出：{relative_path} - "
                        f"{output.get('ename', 'UnknownError')}"
                    )
    return notebook_count, output_count


def validate_personal_paths(files: list[str], errors: list[str]) -> None:
    text_suffixes = {".md", ".py", ".sql", ".yml", ".yaml", ".txt", ".json", ".ipynb"}
    for relative_path in files:
        path = ROOT / relative_path
        if path.suffix.lower() not in text_suffixes:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(pattern.search(text) for pattern in PERSONAL_PATH_PATTERNS):
            errors.append(f"发现本机绝对路径：{relative_path}")


def main() -> None:
    errors: list[str] = []
    files = tracked_files()

    validate_required_paths(errors)
    validate_tracked_scope(files, errors)
    validate_markdown_links(files, errors)
    notebook_count, output_count = validate_notebooks(files, errors)
    validate_personal_paths(files, errors)

    if not compileall.compile_dir(ROOT / "scripts", quiet=1):
        errors.append("scripts/ 中存在无法编译的 Python 文件。")

    if errors:
        print("REPOSITORY AUDIT FAILED")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("REPOSITORY AUDIT PASSED")
    print(f"tracked_files={len(files)}")
    print(f"notebooks={notebook_count}")
    print(f"notebook_outputs={output_count}")


if __name__ == "__main__":
    main()
