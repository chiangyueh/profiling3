import ctypes
import hashlib
import math
import os
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np

from .. import inputs as cost

ROOT = Path(__file__).resolve().parents[2]


def graph_engine():
    source = Path(__file__).with_name("fa_latency_eval.cpp")
    signature = cost.sha256(
        {"source": source.read_text(), "machine": os.uname().machine}
    )
    folder = ROOT / "out" / "fa_latency_host"
    folder.mkdir(parents=True, exist_ok=True)
    library, marker = folder / "latency_eval.so", folder / "latency_eval_signature"
    if not library.is_file() or not marker.is_file() or marker.read_text() != signature:
        result = subprocess.run(
            [
                "g++",
                "-O2",
                "-std=c++17",
                "-shared",
                "-fPIC",
                str(source),
                "-o",
                str(library) + ".tmp",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=120,
        )
        if result.returncode:
            raise RuntimeError(
                "CPU latency evaluator build failed:\n" + result.stdout[-2000:]
            )
        Path(str(library) + ".tmp").replace(library)
        marker.write_text(signature)
    engine = ctypes.CDLL(str(library))
    engine.fa_latency_order.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    engine.fa_latency_order.restype = ctypes.c_int
    engine.fa_latency_eval.argtypes = [ctypes.c_int] * 4 + [ctypes.c_void_p] * 6
    engine.fa_latency_eval.restype = ctypes.c_double
    return engine


def compact_engine():
    cpp = Path(__file__).with_name("fa_branch_graph.cpp")
    folder = ROOT / "out/fa_branch_host"
    folder.mkdir(parents=True, exist_ok=True)
    library = folder / "graph_compact.so"
    marker = folder / "graph_compact.signature"
    key = hashlib.sha256(cpp.read_bytes()).hexdigest() + os.uname().machine
    if not library.is_file() or not marker.is_file() or marker.read_text() != key:
        result = subprocess.run(
            [
                "g++",
                "-O2",
                "-std=c++17",
                "-shared",
                "-fPIC",
                str(cpp),
                "-o",
                str(library) + ".tmp",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=60,
        )
        if result.returncode:
            raise RuntimeError(
                "CPU graph compactor build failed: " + result.stdout[-1000:]
            )
        Path(str(library) + ".tmp").replace(library)
        marker.write_text(key)
    engine = ctypes.CDLL(str(library))
    engine.fa_compact_graph.argtypes = [ctypes.c_int] * 2 + [ctypes.c_void_p] * 8
    engine.fa_compact_graph.restype = ctypes.c_int
    return engine


def compact_graph(graph):
    n = len(graph["a"])
    zero = np.ascontiguousarray(np.all(graph["templates"] == 0, axis=1), dtype=np.uint8)
    a, b, labels = (np.empty(n, dtype=np.int32) for _ in range(3))
    root = ctypes.c_int()
    count = compact_engine().fa_compact_graph(
        n,
        graph["root"],
        graph["a"].ctypes.data,
        graph["b"].ctypes.data,
        graph["which"].ctypes.data,
        zero.ctypes.data,
        a.ctypes.data,
        b.ctypes.data,
        labels.ctypes.data,
        ctypes.byref(root),
    )
    if count <= 0 or root.value < 0:
        raise ValueError("dependency graph compaction failed")
    return {
        **graph,
        "a": a[:count].copy(),
        "b": b[:count].copy(),
        "which": labels[:count].copy(),
        "root": root.value,
    }


def graph_time(graph, rates, gradient=False):
    coefficients = np.ascontiguousarray(rates, dtype=np.float64)
    derivative = np.empty(len(coefficients), dtype=np.float64) if gradient else None
    value = graph_engine().fa_latency_eval(
        len(graph["a"]),
        len(graph["templates"]),
        len(coefficients),
        graph["root"],
        graph["a"].ctypes.data,
        graph["b"].ctypes.data,
        graph["which"].ctypes.data,
        graph["templates"].ctypes.data,
        coefficients.ctypes.data,
        derivative.ctypes.data if gradient else None,
    )
    if not math.isfinite(value):
        raise RuntimeError("source timing graph produced a non-finite duration")
    return (value, derivative) if gradient else value
