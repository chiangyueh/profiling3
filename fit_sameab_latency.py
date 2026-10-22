from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np


DATASET_VERSION = "sameab_latency_v4"
MODEL_VERSION = "sameab_latency_model_v1"
SHAPE_FIELDS = ("B", "N1", "N2", "S1", "S2", "D", "DV", "dtype", "layout")
CONFIG_FIELDS = (
    "requested_s1_base",
    "requested_s2_base",
    "requested_n_ratio",
    "requested_bmm_preset",
)


def as_float(row: dict[str, str], name: str) -> float:
    value = float(row[name])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {name} in {row.get('case', 'unknown case')}")
    return value


def load_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows = []
    for path in paths:
        with path.open(newline="") as source:
            rows.extend(
                row
                for row in csv.DictReader(source)
                if row.get("version") == DATASET_VERSION
            )
    if not rows:
        raise RuntimeError(f"no {DATASET_VERSION} rows found")
    identities = set()
    for row in rows:
        identity = (
            row["split"],
            row["case"],
            row["record_type"],
            row["round"],
            row["config_id"],
        )
        if identity in identities:
            raise RuntimeError(f"duplicate record: {identity}")
        identities.add(identity)
    return rows


def config_of(row: dict[str, str]) -> tuple[int, int, int, int]:
    return tuple(int(row[name]) for name in CONFIG_FIELDS)


def config_key(config: tuple[int, int, int, int]) -> str:
    return ",".join(str(value) for value in config)


def interpolate_baseline(
    candidate: dict[str, str],
    start: dict[str, str],
    end: dict[str, str],
) -> float:
    start_order = as_float(start, "order")
    end_order = as_float(end, "order")
    if end_order <= start_order:
        raise RuntimeError(f"invalid baseline orders for {candidate['case']}")
    position = (as_float(candidate, "order") - start_order) / (end_order - start_order)
    return as_float(start, "duration_us") + position * (
        as_float(end, "duration_us") - as_float(start, "duration_us")
    )


def prepare(
    rows: list[dict[str, str]], require_calibration_holdout: bool = True
) -> tuple[list[dict], list[dict], dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["case"]].append(row)
    cases = []
    observations = []
    status_counts = defaultdict(int)
    for row in rows:
        status_counts[row["status"]] += 1
    for case_name, case_rows in grouped.items():
        split_values = {row["split"] for row in case_rows}
        if len(split_values) != 1:
            raise RuntimeError(f"case appears in multiple splits: {case_name}")
        split = next(iter(split_values))
        first = case_rows[0]
        baselines = [
            row
            for row in case_rows
            if row["record_type"] == "baseline" and row["status"] == "ok"
        ]
        rounds = sorted({row["round"] for row in case_rows})
        if len(baselines) != 2 * len(rounds):
            raise RuntimeError(f"incomplete baselines for {case_name}")
        by_round = {}
        for round_id in rounds:
            round_baselines = [row for row in baselines if row["round"] == round_id]
            starts = [row for row in round_baselines if "start" in row["config_id"]]
            ends = [row for row in round_baselines if "end" in row["config_id"]]
            if len(starts) != 1 or len(ends) != 1:
                raise RuntimeError(f"invalid baseline pair for {case_name} round {round_id}")
            by_round[round_id] = (starts[0], ends[0])
        baseline_us = statistics.median(as_float(row, "duration_us") for row in baselines)
        work = as_float(first, "bmm1_flops") + as_float(first, "bmm2_flops")
        case = {
            "case": case_name,
            "split": split,
            "layout": first["layout"],
            "shape": {name: first[name] for name in SHAPE_FIELDS},
            "work_flops": work,
            "baseline_us": baseline_us,
        }
        cases.append(case)
        valid_candidates = [
            row
            for row in case_rows
            if row["record_type"] == "candidate"
            and row["status"] == "ok"
            and row["matched"] == "True"
            and row["correct"] == "True"
        ]
        candidate_groups = defaultdict(list)
        for row in valid_candidates:
            candidate_groups[config_of(row)].append(row)
        for config, candidate_rows in candidate_groups.items():
            candidate_rounds = {row["round"] for row in candidate_rows}
            if candidate_rounds != set(rounds):
                raise RuntimeError(
                    f"incomplete candidate {config_key(config)} for {case_name}"
                )
            ratios = []
            durations = []
            ids = set()
            for row in candidate_rows:
                start, end = by_round[row["round"]]
                reference = interpolate_baseline(row, start, end)
                ratios.append(as_float(row, "duration_us") / reference)
                durations.append(as_float(row, "duration_us"))
                ids.add(row["config_id"])
            ratio = statistics.median(ratios)
            observations.append(
                {
                    "case": case_name,
                    "split": split,
                    "layout": first["layout"],
                    "config": config,
                    "config_ids": sorted(ids),
                    "work_flops": work,
                    "baseline_us": baseline_us,
                    "actual_ratio": ratio,
                    "actual_adjusted_us": baseline_us * ratio,
                    "actual_raw_us": statistics.median(durations),
                }
            )
    calibration_cases = {case["case"] for case in cases if case["split"] == "calibration"}
    holdout_cases = {case["case"] for case in cases if case["split"] == "holdout"}
    if require_calibration_holdout and (not calibration_cases or not holdout_cases):
        raise RuntimeError("both calibration and holdout cases are required")
    if calibration_cases & holdout_cases:
        raise RuntimeError("calibration and holdout cases overlap")
    return cases, observations, dict(status_counts)


def fit_baseline(cases: list[dict]) -> dict[str, float]:
    grouped = defaultdict(list)
    for case in cases:
        grouped[case["layout"]].append(case)
    slopes = {}
    for layout, items in grouped.items():
        denominator = sum(item["work_flops"] ** 2 for item in items)
        if denominator == 0:
            raise RuntimeError(f"zero work for layout {layout}")
        slopes[layout] = sum(
            item["work_flops"] * item["baseline_us"] for item in items
        ) / denominator
    return slopes


def fit_ratios(observations: list[dict]) -> dict[tuple[str, tuple[int, int, int, int]], float]:
    grouped = defaultdict(list)
    for observation in observations:
        grouped[(observation["layout"], observation["config"])].append(
            observation["actual_ratio"]
        )
    return {key: statistics.median(values) for key, values in grouped.items()}


def predict(
    observation: dict,
    slopes: dict[str, float],
    ratios: dict[tuple[str, tuple[int, int, int, int]], float],
) -> dict:
    layout = observation["layout"]
    ratio_key = (layout, observation["config"])
    if layout not in slopes or ratio_key not in ratios:
        raise KeyError(f"model has no entry for {layout} {config_key(observation['config'])}")
    baseline = slopes[layout] * observation["work_flops"]
    ratio = ratios[ratio_key]
    return {
        "predicted_baseline_us": baseline,
        "predicted_ratio": ratio,
        "predicted_us": baseline * ratio,
    }


def error_metrics(
    actual: list[float], predicted: list[float], rmse_name: str = "rmse_us"
) -> dict:
    actual_values = np.asarray(actual, dtype=np.float64)
    predicted_values = np.asarray(predicted, dtype=np.float64)
    relative = np.abs(predicted_values - actual_values) / actual_values
    return {
        "count": int(actual_values.size),
        "mape_percent": float(np.mean(relative) * 100.0),
        "median_ape_percent": float(np.median(relative) * 100.0),
        "p90_ape_percent": float(np.percentile(relative, 90) * 100.0),
        "max_ape_percent": float(np.max(relative) * 100.0),
        rmse_name: float(np.sqrt(np.mean((predicted_values - actual_values) ** 2))),
        "mean_signed_error_percent": float(
            np.mean((predicted_values - actual_values) / actual_values) * 100.0
        ),
    }


def rank_correlation(actual: list[float], predicted: list[float]) -> float:
    if len(actual) < 2:
        return 1.0
    actual_order = np.argsort(np.asarray(actual, dtype=np.float64), kind="stable")
    predicted_order = np.argsort(np.asarray(predicted, dtype=np.float64), kind="stable")
    actual_rank = np.empty(len(actual_order), dtype=np.float64)
    predicted_rank = np.empty(len(predicted_order), dtype=np.float64)
    actual_rank[actual_order] = np.arange(len(actual_order), dtype=np.float64)
    predicted_rank[predicted_order] = np.arange(len(predicted_order), dtype=np.float64)
    return float(np.corrcoef(actual_rank, predicted_rank)[0, 1])


def evaluate(observations: list[dict], slopes: dict, ratios: dict) -> tuple[dict, list[dict]]:
    predictions = []
    for observation in observations:
        values = predict(observation, slopes, ratios)
        predictions.append({**observation, **values})
    adjusted = error_metrics(
        [item["actual_adjusted_us"] for item in predictions],
        [item["predicted_us"] for item in predictions],
    )
    raw = error_metrics(
        [item["actual_raw_us"] for item in predictions],
        [item["predicted_us"] for item in predictions],
    )
    ratio = error_metrics(
        [item["actual_ratio"] for item in predictions],
        [item["predicted_ratio"] for item in predictions],
        "rmse_ratio",
    )
    by_case = {}
    for case_name in sorted({item["case"] for item in predictions}):
        items = [item for item in predictions if item["case"] == case_name]
        actual = [item["actual_adjusted_us"] for item in items]
        predicted = [item["predicted_us"] for item in items]
        actual_best = min(items, key=lambda item: item["actual_adjusted_us"])
        predicted_best = min(items, key=lambda item: item["predicted_us"])
        by_case[case_name] = {
            "layout": items[0]["layout"],
            "adjusted_latency": error_metrics(actual, predicted),
            "raw_latency": error_metrics(
                [item["actual_raw_us"] for item in items], predicted
            ),
            "ratio": error_metrics(
                [item["actual_ratio"] for item in items],
                [item["predicted_ratio"] for item in items],
                "rmse_ratio",
            ),
            "rank_correlation": rank_correlation(actual, predicted),
            "actual_best_config": config_key(actual_best["config"]),
            "predicted_best_config": config_key(predicted_best["config"]),
            "selection_regret_percent": (
                predicted_best["actual_adjusted_us"]
                / actual_best["actual_adjusted_us"]
                - 1.0
            )
            * 100.0,
        }
    return {
        "adjusted_latency": adjusted,
        "raw_latency": raw,
        "ratio": ratio,
        "by_case": by_case,
    }, predictions


def grouped_cross_validation(cases: list[dict], observations: list[dict]) -> tuple[dict, list[dict]]:
    predictions = []
    for held_out in sorted(case["case"] for case in cases):
        train_cases = [case for case in cases if case["case"] != held_out]
        train_observations = [item for item in observations if item["case"] != held_out]
        test_observations = [item for item in observations if item["case"] == held_out]
        slopes = fit_baseline(train_cases)
        ratios = fit_ratios(train_observations)
        for observation in test_observations:
            predictions.append({**observation, **predict(observation, slopes, ratios)})
    metrics, _ = evaluate_from_predictions(predictions)
    return metrics, predictions


def evaluate_from_predictions(predictions: list[dict]) -> tuple[dict, list[dict]]:
    adjusted = error_metrics(
        [item["actual_adjusted_us"] for item in predictions],
        [item["predicted_us"] for item in predictions],
    )
    raw = error_metrics(
        [item["actual_raw_us"] for item in predictions],
        [item["predicted_us"] for item in predictions],
    )
    ratio = error_metrics(
        [item["actual_ratio"] for item in predictions],
        [item["predicted_ratio"] for item in predictions],
        "rmse_ratio",
    )
    return {
        "adjusted_latency": adjusted,
        "raw_latency": raw,
        "ratio": ratio,
    }, predictions


def baseline_evaluation(cases: list[dict], slopes: dict[str, float]) -> dict:
    return error_metrics(
        [case["baseline_us"] for case in cases],
        [slopes[case["layout"]] * case["work_flops"] for case in cases],
    )


def write_predictions(path: Path, predictions: list[dict]) -> None:
    fields = (
        "split",
        "case",
        "layout",
        *CONFIG_FIELDS,
        "actual_raw_us",
        "actual_adjusted_us",
        "predicted_us",
        "actual_ratio",
        "predicted_ratio",
        "predicted_baseline_us",
        "adjusted_ape_percent",
        "raw_ape_percent",
    )
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for item in predictions:
            config = item["config"]
            writer.writerow(
                {
                    "split": item["split"],
                    "case": item["case"],
                    "layout": item["layout"],
                    **dict(zip(CONFIG_FIELDS, config)),
                    "actual_raw_us": item["actual_raw_us"],
                    "actual_adjusted_us": item["actual_adjusted_us"],
                    "predicted_us": item["predicted_us"],
                    "actual_ratio": item["actual_ratio"],
                    "predicted_ratio": item["predicted_ratio"],
                    "predicted_baseline_us": item["predicted_baseline_us"],
                    "adjusted_ape_percent": abs(
                        item["predicted_us"] - item["actual_adjusted_us"]
                    )
                    / item["actual_adjusted_us"]
                    * 100.0,
                    "raw_ape_percent": abs(
                        item["predicted_us"] - item["actual_raw_us"]
                    )
                    / item["actual_raw_us"]
                    * 100.0,
                }
            )


def serialize_model(slopes: dict, ratios: dict) -> dict:
    return {
        "model_version": MODEL_VERSION,
        "dataset_version": DATASET_VERSION,
        "scope": {
            "operator": "FlashAttentionScore",
            "route": "SameAB",
            "dtype": "BF16",
            "D": 192,
            "DV": 128,
            "layouts": sorted(slopes),
        },
        "baseline_model": {
            "type": "total_flops_through_origin_by_layout",
            "formula": "baseline_us = coefficient_us_per_flop[layout] * (bmm1_flops + bmm2_flops)",
            "coefficient_us_per_flop": slopes,
        },
        "ratio_model": {
            "type": "median_ratio_by_layout_and_tiling",
            "formula": "candidate_us = baseline_us * ratio",
            "entries": [
                {
                    "layout": layout,
                    **dict(zip(CONFIG_FIELDS, config)),
                    "ratio": value,
                }
                for (layout, config), value in sorted(
                    ratios.items(), key=lambda item: (item[0][0], item[0][1])
                )
            ],
        },
        "unsupported": [
            "layouts, dtypes, D, or DV outside the declared scope",
            "tiling tuples absent from ratio_model.entries",
            "invalid configurations rejected by the validator",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output") / MODEL_VERSION,
    )
    parser.add_argument("--external-csv", nargs="*", type=Path, default=[])
    args = parser.parse_args()
    rows = load_rows(args.csv)
    cases, observations, status_counts = prepare(rows)
    calibration_cases = [case for case in cases if case["split"] == "calibration"]
    holdout_cases = [case for case in cases if case["split"] == "holdout"]
    calibration = [item for item in observations if item["split"] == "calibration"]
    holdout = [item for item in observations if item["split"] == "holdout"]
    cv_metrics, cv_predictions = grouped_cross_validation(calibration_cases, calibration)
    slopes = fit_baseline(calibration_cases)
    ratios = fit_ratios(calibration)
    holdout_metrics, holdout_predictions = evaluate(holdout, slopes, ratios)
    report = {
        "model_version": MODEL_VERSION,
        "dataset_version": DATASET_VERSION,
        "data": {
            "rows": len(rows),
            "status_counts": status_counts,
            "calibration_cases": len(calibration_cases),
            "holdout_cases": len(holdout_cases),
            "calibration_observations": len(calibration),
            "holdout_observations": len(holdout),
        },
        "calibration_leave_one_shape_out": cv_metrics,
        "holdout": {
            "baseline": baseline_evaluation(holdout_cases, slopes),
            **holdout_metrics,
        },
    }
    external_predictions = []
    if args.external_csv:
        external_rows = load_rows(args.external_csv)
        external_cases, external_observations, external_status_counts = prepare(
            external_rows, require_calibration_holdout=False
        )
        if any(case["split"] != "external" for case in external_cases):
            raise RuntimeError("external evaluation contains a non-external split")
        external_metrics, external_predictions = evaluate(
            external_observations, slopes, ratios
        )
        report["external"] = {
            "data": {
                "rows": len(external_rows),
                "status_counts": external_status_counts,
                "cases": len(external_cases),
                "observations": len(external_observations),
            },
            "baseline": baseline_evaluation(external_cases, slopes),
            **external_metrics,
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "model.json").write_text(
        json.dumps(serialize_model(slopes, ratios), indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    write_predictions(args.output_dir / "calibration_cv_predictions.csv", cv_predictions)
    write_predictions(args.output_dir / "holdout_predictions.csv", holdout_predictions)
    if external_predictions:
        write_predictions(args.output_dir / "external_predictions.csv", external_predictions)
    print(f"MODEL={args.output_dir / 'model.json'}")
    print(f"REPORT={args.output_dir / 'report.json'}")
    print(
        "CALIBRATION_LOSO "
        f"MAPE={cv_metrics['adjusted_latency']['mape_percent']:.3f}% "
        f"MEDIAN_APE={cv_metrics['adjusted_latency']['median_ape_percent']:.3f}%"
    )
    print(
        "HOLDOUT_ADJUSTED "
        f"MAPE={holdout_metrics['adjusted_latency']['mape_percent']:.3f}% "
        f"MEDIAN_APE={holdout_metrics['adjusted_latency']['median_ape_percent']:.3f}% "
        f"P90_APE={holdout_metrics['adjusted_latency']['p90_ape_percent']:.3f}% "
        f"MAX_APE={holdout_metrics['adjusted_latency']['max_ape_percent']:.3f}%"
    )
    print(
        "HOLDOUT_RAW "
        f"MAPE={holdout_metrics['raw_latency']['mape_percent']:.3f}% "
        f"MEDIAN_APE={holdout_metrics['raw_latency']['median_ape_percent']:.3f}%"
    )
    for case_name, metrics in holdout_metrics["by_case"].items():
        print(
            f"CASE={case_name} "
            f"MAPE={metrics['adjusted_latency']['mape_percent']:.3f}% "
            f"RANK={metrics['rank_correlation']:.4f} "
            f"REGRET={metrics['selection_regret_percent']:.3f}% "
            f"PREDICTED_BEST={metrics['predicted_best_config']} "
            f"ACTUAL_BEST={metrics['actual_best_config']}"
        )
    if "external" in report:
        external = report["external"]
        print(
            "EXTERNAL_ADJUSTED "
            f"MAPE={external['adjusted_latency']['mape_percent']:.3f}% "
            f"MEDIAN_APE={external['adjusted_latency']['median_ape_percent']:.3f}% "
            f"P90_APE={external['adjusted_latency']['p90_ape_percent']:.3f}% "
            f"MAX_APE={external['adjusted_latency']['max_ape_percent']:.3f}%"
        )
        for case_name, metrics in external["by_case"].items():
            print(
                f"EXTERNAL_CASE={case_name} "
                f"MAPE={metrics['adjusted_latency']['mape_percent']:.3f}% "
                f"RANK={metrics['rank_correlation']:.4f} "
                f"REGRET={metrics['selection_regret_percent']:.3f}% "
                f"PREDICTED_BEST={metrics['predicted_best_config']} "
                f"ACTUAL_BEST={metrics['actual_best_config']}"
            )


if __name__ == "__main__":
    main()
