import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"report is not a JSON object: {path}")
    return data


def _maybe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _delta(current: Optional[float], baseline: Optional[float]) -> Optional[float]:
    if current is None or baseline is None:
        return None
    return round(current - baseline, 4)


def _index_results(report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    rows = report.get("results", [])
    if not isinstance(rows, list):
        return {}
    output: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        sample_id = str(row.get("sample_id", "")).strip()
        if sample_id:
            output[sample_id] = row
    return output


def compare_reports(current: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
    current_summary = current.get("summary", {}) if isinstance(current.get("summary", {}), dict) else {}
    baseline_summary = baseline.get("summary", {}) if isinstance(baseline.get("summary", {}), dict) else {}

    current_fact = _maybe_float(current_summary.get("fact_avg_score"))
    baseline_fact = _maybe_float(baseline_summary.get("fact_avg_score"))
    current_task = _maybe_float(current_summary.get("task_avg_score"))
    baseline_task = _maybe_float(baseline_summary.get("task_avg_score"))
    current_fact_confidence = _maybe_float(current_summary.get("fact_judge_confidence_avg"))
    baseline_fact_confidence = _maybe_float(baseline_summary.get("fact_judge_confidence_avg"))
    current_task_confidence = _maybe_float(current_summary.get("task_judge_confidence_avg"))
    baseline_task_confidence = _maybe_float(baseline_summary.get("task_judge_confidence_avg"))
    current_support_ratio = _maybe_float(current_summary.get("support_ratio_avg"))
    baseline_support_ratio = _maybe_float(baseline_summary.get("support_ratio_avg"))
    current_task_coverage = _maybe_float(current_summary.get("task_coverage_ratio_avg"))
    baseline_task_coverage = _maybe_float(baseline_summary.get("task_coverage_ratio_avg"))
    current_hall_density = _maybe_float(current_summary.get("hallucination_density_avg"))
    baseline_hall_density = _maybe_float(baseline_summary.get("hallucination_density_avg"))
    current_weighted_hall_density = _maybe_float(current_summary.get("weighted_hallucination_density_avg"))
    baseline_weighted_hall_density = _maybe_float(baseline_summary.get("weighted_hallucination_density_avg"))
    current_total_claim_importance = _maybe_float(current_summary.get("total_claim_importance"))
    baseline_total_claim_importance = _maybe_float(baseline_summary.get("total_claim_importance"))
    current_total_requirement_importance = _maybe_float(current_summary.get("total_requirement_importance"))
    baseline_total_requirement_importance = _maybe_float(baseline_summary.get("total_requirement_importance"))

    current_rows = _index_results(current)
    baseline_rows = _index_results(baseline)

    sample_deltas = []
    for sample_id, current_row in current_rows.items():
        baseline_row = baseline_rows.get(sample_id)
        if baseline_row is None:
            continue

        c_fact = _maybe_float(current_row.get("fact_score"))
        b_fact = _maybe_float(baseline_row.get("fact_score"))
        c_task = _maybe_float(current_row.get("task_score"))
        b_task = _maybe_float(baseline_row.get("task_score"))
        c_fact_confidence = _maybe_float(current_row.get("fact_judge_confidence"))
        b_fact_confidence = _maybe_float(baseline_row.get("fact_judge_confidence"))
        c_hall_density = _maybe_float(current_row.get("hallucination_density"))
        b_hall_density = _maybe_float(baseline_row.get("hallucination_density"))
        c_weighted_hall_density = _maybe_float(current_row.get("weighted_hallucination_density"))
        b_weighted_hall_density = _maybe_float(baseline_row.get("weighted_hallucination_density"))
        c_support_ratio = _maybe_float(current_row.get("support_ratio"))
        b_support_ratio = _maybe_float(baseline_row.get("support_ratio"))
        c_task_coverage = _maybe_float(current_row.get("task_coverage_ratio"))
        b_task_coverage = _maybe_float(baseline_row.get("task_coverage_ratio"))

        fact_delta = _delta(c_fact, b_fact)
        task_delta = _delta(c_task, b_task)
        fact_confidence_delta = _delta(c_fact_confidence, b_fact_confidence)
        hall_density_delta = _delta(c_hall_density, b_hall_density)
        weighted_hall_density_delta = _delta(c_weighted_hall_density, b_weighted_hall_density)
        support_ratio_delta = _delta(c_support_ratio, b_support_ratio)
        task_coverage_delta = _delta(c_task_coverage, b_task_coverage)

        if (
            fact_delta == 0
            and task_delta == 0
            and fact_confidence_delta == 0
            and hall_density_delta == 0
            and weighted_hall_density_delta == 0
            and support_ratio_delta == 0
            and task_coverage_delta == 0
        ):
            continue

        sample_deltas.append(
            {
                "sample_id": sample_id,
                "fact_delta": fact_delta,
                "task_delta": task_delta,
                "fact_confidence_delta": fact_confidence_delta,
                "hallucination_density_delta": hall_density_delta,
                "weighted_hallucination_density_delta": weighted_hall_density_delta,
                "support_ratio_delta": support_ratio_delta,
                "task_coverage_ratio_delta": task_coverage_delta,
                "current_fact": c_fact,
                "baseline_fact": b_fact,
                "current_task": c_task,
                "baseline_task": b_task,
                "current_fact_confidence": c_fact_confidence,
                "baseline_fact_confidence": b_fact_confidence,
                "current_hallucination_density": c_hall_density,
                "baseline_hallucination_density": b_hall_density,
                "current_weighted_hallucination_density": c_weighted_hall_density,
                "baseline_weighted_hallucination_density": b_weighted_hall_density,
                "current_support_ratio": c_support_ratio,
                "baseline_support_ratio": b_support_ratio,
                "current_task_coverage_ratio": c_task_coverage,
                "baseline_task_coverage_ratio": b_task_coverage,
                "current_output": str(current_row.get("final_summary_path", "")),
            }
        )

    return {
        "generated_at": datetime.now().isoformat(),
        "current_run_id": current.get("run_id", "unknown"),
        "baseline_run_id": baseline.get("run_id", "unknown"),
        "overview": {
            "current_fact_avg": current_fact,
            "baseline_fact_avg": baseline_fact,
            "fact_delta": _delta(current_fact, baseline_fact),
            "current_task_avg": current_task,
            "baseline_task_avg": baseline_task,
            "task_delta": _delta(current_task, baseline_task),
            "current_fact_confidence_avg": current_fact_confidence,
            "baseline_fact_confidence_avg": baseline_fact_confidence,
            "fact_confidence_delta": _delta(current_fact_confidence, baseline_fact_confidence),
            "current_task_confidence_avg": current_task_confidence,
            "baseline_task_confidence_avg": baseline_task_confidence,
            "task_confidence_delta": _delta(current_task_confidence, baseline_task_confidence),
            "current_support_ratio_avg": current_support_ratio,
            "baseline_support_ratio_avg": baseline_support_ratio,
            "support_ratio_delta": _delta(current_support_ratio, baseline_support_ratio),
            "current_task_coverage_ratio_avg": current_task_coverage,
            "baseline_task_coverage_ratio_avg": baseline_task_coverage,
            "task_coverage_ratio_delta": _delta(current_task_coverage, baseline_task_coverage),
            "current_hallucination_density_avg": current_hall_density,
            "baseline_hallucination_density_avg": baseline_hall_density,
            "hallucination_density_delta": _delta(current_hall_density, baseline_hall_density),
            "current_weighted_hallucination_density_avg": current_weighted_hall_density,
            "baseline_weighted_hallucination_density_avg": baseline_weighted_hall_density,
            "weighted_hallucination_density_delta": _delta(
                current_weighted_hall_density, baseline_weighted_hall_density
            ),
            "current_total_claim_importance": current_total_claim_importance,
            "baseline_total_claim_importance": baseline_total_claim_importance,
            "total_claim_importance_delta": _delta(current_total_claim_importance, baseline_total_claim_importance),
            "current_total_requirement_importance": current_total_requirement_importance,
            "baseline_total_requirement_importance": baseline_total_requirement_importance,
            "total_requirement_importance_delta": _delta(
                current_total_requirement_importance, baseline_total_requirement_importance
            ),
        },
        "sample_deltas": sample_deltas,
    }


def render_markdown(diff_report: Dict[str, Any]) -> str:
    overview = diff_report.get("overview", {})
    lines = [
        "# Evaluation Baseline Comparison",
        "",
        f"- generated_at: {diff_report.get('generated_at')}",
        f"- current_run_id: {diff_report.get('current_run_id')}",
        f"- baseline_run_id: {diff_report.get('baseline_run_id')}",
        "",
        "## Overview",
        "",
        f"- current_fact_avg: {overview.get('current_fact_avg')}",
        f"- baseline_fact_avg: {overview.get('baseline_fact_avg')}",
        f"- fact_delta: {overview.get('fact_delta')}",
        f"- current_task_avg: {overview.get('current_task_avg')}",
        f"- baseline_task_avg: {overview.get('baseline_task_avg')}",
        f"- task_delta: {overview.get('task_delta')}",
        f"- current_fact_confidence_avg: {overview.get('current_fact_confidence_avg')}",
        f"- baseline_fact_confidence_avg: {overview.get('baseline_fact_confidence_avg')}",
        f"- fact_confidence_delta: {overview.get('fact_confidence_delta')}",
        f"- current_task_confidence_avg: {overview.get('current_task_confidence_avg')}",
        f"- baseline_task_confidence_avg: {overview.get('baseline_task_confidence_avg')}",
        f"- task_confidence_delta: {overview.get('task_confidence_delta')}",
        f"- current_support_ratio_avg: {overview.get('current_support_ratio_avg')}",
        f"- baseline_support_ratio_avg: {overview.get('baseline_support_ratio_avg')}",
        f"- support_ratio_delta: {overview.get('support_ratio_delta')}",
        f"- current_task_coverage_ratio_avg: {overview.get('current_task_coverage_ratio_avg')}",
        f"- baseline_task_coverage_ratio_avg: {overview.get('baseline_task_coverage_ratio_avg')}",
        f"- task_coverage_ratio_delta: {overview.get('task_coverage_ratio_delta')}",
        f"- current_hallucination_density_avg: {overview.get('current_hallucination_density_avg')}",
        f"- baseline_hallucination_density_avg: {overview.get('baseline_hallucination_density_avg')}",
        f"- hallucination_density_delta: {overview.get('hallucination_density_delta')}",
        f"- current_weighted_hallucination_density_avg: {overview.get('current_weighted_hallucination_density_avg')}",
        f"- baseline_weighted_hallucination_density_avg: {overview.get('baseline_weighted_hallucination_density_avg')}",
        f"- weighted_hallucination_density_delta: {overview.get('weighted_hallucination_density_delta')}",
        f"- current_total_claim_importance: {overview.get('current_total_claim_importance')}",
        f"- baseline_total_claim_importance: {overview.get('baseline_total_claim_importance')}",
        f"- total_claim_importance_delta: {overview.get('total_claim_importance_delta')}",
        f"- current_total_requirement_importance: {overview.get('current_total_requirement_importance')}",
        f"- baseline_total_requirement_importance: {overview.get('baseline_total_requirement_importance')}",
        f"- total_requirement_importance_delta: {overview.get('total_requirement_importance_delta')}",
        "",
        "## Per Sample Deltas",
        "",
    ]

    rows = diff_report.get("sample_deltas", [])
    if not rows:
        lines.append("- no changed sample deltas")
    else:
        for row in rows:
            lines.append(
                f"- {row.get('sample_id')}: fact_delta={row.get('fact_delta')} task_delta={row.get('task_delta')} fact_confidence_delta={row.get('fact_confidence_delta')} support_ratio_delta={row.get('support_ratio_delta')} task_coverage_ratio_delta={row.get('task_coverage_ratio_delta')} hallucination_density_delta={row.get('hallucination_density_delta')} weighted_hallucination_density_delta={row.get('weighted_hallucination_density_delta')} output={row.get('current_output')}"
            )

    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two evaluation report.json files")
    parser.add_argument("--current", required=True, help="Path to current report.json")
    parser.add_argument("--baseline", required=True, help="Path to baseline report.json")
    parser.add_argument(
        "--output",
        default="",
        help="Optional output markdown path. Default writes next to current report as compare_<timestamp>.md",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional output JSON path. Default writes next to current report as compare_<timestamp>.json",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    current_path = (PROJECT_ROOT / args.current).resolve()
    baseline_path = (PROJECT_ROOT / args.baseline).resolve()

    if not current_path.exists():
        raise FileNotFoundError(f"current report not found: {current_path}")
    if not baseline_path.exists():
        raise FileNotFoundError(f"baseline report not found: {baseline_path}")

    current = _load_json(current_path)
    baseline = _load_json(baseline_path)
    diff_report = compare_reports(current, baseline)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_md = current_path.parent / f"compare_{ts}.md"
    default_json = current_path.parent / f"compare_{ts}.json"

    output_md = (PROJECT_ROOT / args.output).resolve() if args.output.strip() else default_md
    output_json = (PROJECT_ROOT / args.output_json).resolve() if args.output_json.strip() else default_json

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    with output_md.open("w", encoding="utf-8") as f:
        f.write(render_markdown(diff_report))

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(diff_report, f, ensure_ascii=False, indent=2)

    print(f"done: {output_md}")
    print(f"done: {output_json}")


if __name__ == "__main__":
    main()
