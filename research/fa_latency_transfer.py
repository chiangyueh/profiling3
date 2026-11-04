from __future__ import annotations
import copy
import hashlib
import math
from collections import Counter
from pathlib import Path
import numpy as np
from research import fa_latency_refinement as micro
from prediction.fa_latency_transfer import (
    signature,
    api_kernel_inputs,
    transfer,
    Work,
    workload,
    features,
    Graph,
    branch_graph,
    predict,
)
BASE_FEATURES = ("payload_kb", "copy_calls", "strided_rows")
EXTRA_FEATURES = ("nd2nz_partial_rows", "nd2nz_split_sector_kb", "capacity_pressure_kb")
FEATURES = BASE_FEATURES + EXTRA_FEATURES
PARAMETERS = micro.parameter_names() + tuple(("mte2", name) for name in FEATURES)
VERSION = "varlen_l1_nd2nz_v4"


def fit(samples, hardware):
    from scipy.optimize import linprog
    selected = [row for row in samples if row["cost_class"].startswith("varlen_l1|") and row["targets"].get("mte2", 0) > 0]
    matrix, targets = [], []
    for row in selected:
        observed = row.get("dma_features") or features(workload(row["inputs"], hardware))
        matrix.append([observed[name] for name in FEATURES])
        targets.append(row["targets"]["mte2"])
    x, y = np.array(matrix, dtype=float), np.array(targets, dtype=float)
    scale = np.maximum(np.max(x / y[:, None], axis=0), 1e-30)
    normalized = x / y[:, None] / scale
    n = len(y)
    result = linprog(np.r_[np.zeros(len(FEATURES)), np.ones(n) / n],
        A_ub=np.vstack((np.c_[normalized, -np.eye(n)], np.c_[-normalized, -np.eye(n)])),
        b_ub=np.r_[np.ones(n), -np.ones(n)], bounds=(0, None), method="highs", options={"threads": 1})
    if not result.success:
        raise RuntimeError("source DMA-cost calibration failed: " + result.message)
    coefficients = result.x[:len(FEATURES)] / scale
    return {"version": VERSION, "source_sha256": signature(), "features": list(FEATURES),
            "coefficients": dict(zip(FEATURES, map(float, coefficients))), "feature_rank": int(np.linalg.matrix_rank(normalized)),
            "training_inputs": sorted(row["input_sha256"] for row in selected), "calibration_samples": n,
            "calibration_target": "original measured whole-kernel AIC MTE2 active-sum microseconds; no kernel latency target",
            "new_validation_latency_used_for_fit": False, "shared_fp16_bf16_dma_cost": True,
            "source": "flash_attention_var_len_score.h CopyInMm1AToL1/CopyInMm1BToL1/CopyInMm2AToL1/CopyInMm2BToL1/CopyGmToL1",
            "stride_semantics": "srcDValue selects the next row; skipped address holes are not payload",
            "reuse_semantics": "Q once per query task; K/V 256-row ping-pong buffers; padded contiguous P; no assumed cross-core L1 or L2 hit",
            "cache_miss_cost_independently_calibrated": False}


def fit_extensions(samples, hardware, previous):
    from scipy.optimize import linprog
    selected = [row for row in samples if row["cost_class"].startswith("varlen_l1|") and row["targets"].get("mte2", 0) > 0]
    matrix, targets, baseline = [], [], []
    for row in selected:
        values = row.get("dma_features") or features(workload(row["inputs"], hardware))
        matrix.append([values[name] for name in EXTRA_FEATURES])
        targets.append(row["targets"]["mte2"])
        baseline.append(sum(values[name] * previous["coefficients"][name] for name in BASE_FEATURES))
    x, y, old = np.array(matrix, dtype=float), np.array(targets, dtype=float), np.array(baseline, dtype=float)
    scale = np.maximum(np.max(x / y[:, None], axis=0), 1e-30)
    normalized = x / y[:, None] / scale
    rank = int(np.linalg.matrix_rank(normalized))
    if rank != len(EXTRA_FEATURES):
        raise ValueError("additional DMA costs are not identifiable from these measurements")
    n = len(y)
    target = (y - old) / y
    result = linprog(np.r_[np.zeros(len(EXTRA_FEATURES)), np.ones(n) / n],
        A_ub=np.vstack((np.c_[normalized, -np.eye(n)], np.c_[-normalized, -np.eye(n)])),
        b_ub=np.r_[target, -target], bounds=(0, None), method="highs", options={"threads": 1})
    if not result.success:
        raise RuntimeError("DMA extension calibration failed: " + result.message)
    rates = result.x[:len(EXTRA_FEATURES)] / scale
    coefficients = {name: previous["coefficients"][name] for name in BASE_FEATURES}
    coefficients.update(zip(EXTRA_FEATURES, map(float, rates)))
    return {**previous, "version": VERSION, "source_sha256": signature(), "features": list(FEATURES),
            "coefficients": coefficients, "extension_feature_rank": rank,
            "extension_training_inputs": sorted(row["input_sha256"] for row in selected),
            "extension_calibration_samples": n,
            "extension_calibration_target": "measured whole-kernel AIC MTE2 active-sum; repeated inputs grouped by median; original three DMA coefficients frozen; kernel latency never fitted",
            "extension_feature_support": {name: sorted({micro.source.cost.sha256({key: row['inputs'][key] for key in ('S1', 'S2')})
                for row, values in zip(selected, x) if values[index] > 0}) for index, name in enumerate(EXTRA_FEATURES)},
            "alignment_reference": "https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/83RC1alpha002/opdevg/ascendcbestP/atlas_ascendc_best_practices_10_10014.html",
            "alignment_semantics": "ND2NZ partial rows plus excess 256-byte sector capacity of rows not aligned to 128 bytes, weighted by complete sectors; geometric efficiency proxy, not additional payload; P is already NZ and excluded",
            "capacity_semantics": "K/V source payload times max(0, 1-L2_bytes/KV_footprint_bytes); capacity-pressure proxy, not an exact cache-hit simulation",
            "end_fence_source": "Matmul Scheduler::End -> TBufPoolL0::ResetCache waits M_MTE1 for both L0 slots; BMM1 End per N chunk, BMM2 End after GetTensorC",
            "cache_miss_cost_independently_calibrated": False,
            "stage_costs_independently_measured": False,
            "prospective_validation_required": True}
