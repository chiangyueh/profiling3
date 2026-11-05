import hashlib
import subprocess
from functools import lru_cache
from pathlib import Path

from .environment import cann_root
from .. import inputs as cost

ROOT = Path(__file__).resolve().parents[2]


def tiler():
    sdk = cann_root()
    cpp = Path(__file__).with_name("fa_branch_matmul_tiling.cpp")
    library = next(
        sdk / "lib64" / name
        for name in ("libtiling_api.so", "libtiling_api.a")
        if (sdk / "lib64" / name).is_file()
    )
    identity = cost.sha256(
        {
            "source": cpp.read_text(),
            "api": hashlib.sha256(library.read_bytes()).hexdigest(),
        }
    )
    folder = ROOT / "out/fa_branch_host"
    folder.mkdir(parents=True, exist_ok=True)
    binary, marker = folder / "matmul_tiling", folder / "signature"
    if not binary.is_file() or not marker.is_file() or marker.read_text() != identity:
        command = [
            "g++",
            "-O0",
            "-std=c++17",
            str(cpp),
            "-I",
            str(sdk / "include"),
            "-I",
            str(sdk / "include/ascendc/host_api"),
            "-L",
            str(sdk / "lib64"),
            "-Wl,-rpath," + str(sdk / "lib64"),
            "-ltiling_api",
            "-lplatform",
            "-lregister",
            "-lascendalog",
            "-lgraph_base",
            "-ldl",
            "-lc_sec",
            "-o",
            str(binary) + ".tmp",
        ]
        built = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=120,
        )
        if built.returncode:
            raise RuntimeError(
                "CPU Matmul trace helper failed: " + built.stdout[-2000:]
            )
        Path(str(binary) + ".tmp").replace(binary)
        marker.write_text(identity)
    return str(binary), identity


def geometry(requests):
    binary, _ = tiler()
    result = subprocess.run(
        [str(binary)],
        input="\n".join(" ".join(map(str, row)) for row in requests) + "\n",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=15,
    )
    names = (
        "base_m",
        "base_n",
        "base_k",
        "depth_a1",
        "depth_b1",
        "db_l0c",
        "step_m",
        "step_n",
        "db_l0a",
        "db_l0b",
        "step_ka",
        "step_kb",
        "iterate_order",
        "trans_length",
    )
    lines = result.stdout.strip().splitlines()
    if (
        result.returncode
        or len(lines) != 2
        or any(len(line.split()) != len(names) for line in lines)
    ):
        raise ValueError(
            "CPU Matmul API rejected branch geometry: " + result.stderr[-500:]
        )
    return [dict(zip(names, map(int, line.split()))) for line in lines]


def mm_requests(p, s, hardware):
    align = cost.align
    base, s2base = s["s1_base"], s["s2_base"]
    same, batch = s["same_ab"], s["family"] in ("b", "b_drop")
    window = (
        min(s2base * s["n_ratio"], p["S2"])
        if same or s["family"] in ("general", "varlen")
        else p["S2"]
    )
    qstride = p["D"] * (
        1
        if p["layout"] == "BNSD"
        else p["N1"] * (p["B"] if p["layout"] == "SBH" else 1)
    )
    kstride = p["D"] * (
        1
        if p["layout"] == "BNSD"
        else p["N2"] * (p["B"] if p["layout"] == "SBH" else 1)
    )
    vstride = kstride // p["D"] * p["DV"]
    l1 = int(hardware["l1_bytes"]) - (
        align(p["S2"]) * align(p["D"]) * 4 if s["family"] == "s1" and s["l1"] else 0
    )
    requests = []
    for stage in (1, 2):
        m, n, k = (
            min(base, p["S1"]),
            window if stage == 1 else p["DV"],
            p["D"] if stage == 1 else window,
        )
        om, on, ka, kb = p["S1"], s2base * s["n_ratio"], qstride, kstride
        fm = fn = fk = -1
        if stage == 2:
            on, ka, kb = vstride, window, vstride
        if batch:
            m = p["S1"]
            om, on = p["S1"], p["S2"] if stage == 1 else p["DV"]
            ka, kb = (qstride, kstride) if stage == 1 else (p["S2"], p["DV"])
            fm, fn = base, (
                s2base
                if stage == 1
                else min(align(p["DV"]), (131072 // (base * 4)) // 16 * 16)
            )
        elif s["family"] == "s1":
            om, on, ka, kb = (
                (p["S1"], p["S2"], qstride, kstride)
                if stage == 1
                else (p["S1"], vstride, p["S2"], vstride)
            )
            fm, fn = min(128, align(p["S1"])), min(
                128, align(p["S2"] if stage == 1 else p["DV"])
            )
            if stage == 2 and p["S1"] == p["S2"] and p["S2"] >= 928 and p["D"] == 64:
                fm, fn, fk = 64, 64, 256
        elif p["layout"] == "TND":
            if stage == 2 and s["l1"]:
                fm, fn = align(min(base, p["S1"])), -1 if same else align(p["DV"])
        elif stage == 1 and (64 < p["D"] <= 128 or s["policy"] == "UNSPLITK"):
            fm, fn, fk = (
                min(128 if same else base, align(p["S1"])),
                128,
                align(p["D"]) if same else p["D"],
            )
        elif stage == 2:
            if same and (64 < p["DV"] <= 128 or s["policy"] == "UNSPLITK"):
                fm, fn = min(128, align(p["S1"])), align(p["DV"])
            if p["DV"] == 64 and p["layout"] == "BNSD" and m >= 64:
                fm, fn, fk = 64, 64, 256
        requests.append(
            (
                m,
                n,
                k,
                om,
                on,
                ka,
                kb,
                fm,
                fn,
                fk,
                l1,
                131072,
                int(hardware["ub_bytes"]),
                2 if p["dtype"] == "BF16" else 0,
                int(batch and stage == 2),
                int(s["family"] == "s1" and stage == 2),
                p["B"],
                p["S1"],
                p["S2"],
                p["N2"],
                p["N1"] // p["N2"],
                stage,
                int(batch),
                int(same),
                int(s["policy"] == "UNSPLITK"),
            )
        )
    return requests
