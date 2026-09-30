from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path


PREFIX = "AUTOTILING_DUMP"


def _bits(value: int, shift: int, width: int) -> int:
    return (value >> shift) & ((1 << width) - 1)


def _read(data: bytes, offset: int, fmt: str) -> int:
    size = struct.calcsize(fmt)
    if offset + size > len(data):
        raise ValueError(f"tiling data is too short for offset {offset} ({len(data)} bytes)")
    return struct.unpack_from("<" + fmt, data, offset)[0]


def decode_fa_key(key: int) -> dict[str, int]:
    return {
        "kernel_type": _bits(key, 0, 4),
        "ub0_axis": _bits(key, 4, 4),
        "ub1_axis": _bits(key, 8, 4),
        "block_axis": _bits(key, 12, 4),
        "impl_mode": _bits(key, 16, 2),
        "dtype": _bits(key, 18, 2),
        "layout": _bits(key, 20, 3),
        "bmm1_format": _bits(key, 23, 1),
        "bmm2_source": _bits(key, 24, 1),
        "sparse": _bits(key, 25, 4),
        "big_double_buffer": _bits(key, 29, 2),
        "has_dropout": _bits(key, 31, 1),
        "has_mask": _bits(key, 32, 1),
        "has_pse": _bits(key, 33, 1),
        "enable_l1_reuse": _bits(key, 34, 1),
        "has_rope": _bits(key, 35, 1),
        "matmul_policy": _bits(key, 36, 1),
        "s1_template": _bits(key, 37, 4),
        "s2_template": _bits(key, 41, 4),
        "d_template": _bits(key, 45, 4),
    }


def decode_fag_key(key: int) -> dict[str, int]:
    return {
        "ub0_axis": _bits(key, 0, 4),
        "ub1_axis": _bits(key, 4, 4),
        "block_axis": _bits(key, 8, 4),
        "same_ab": _bits(key, 12, 1),
        "dtype": _bits(key, 13, 2),
        "layout": _bits(key, 15, 2),
        "sparse": _bits(key, 17, 4),
        "matmul_config": _bits(key, 21, 2),
        "mm12_nd_out": _bits(key, 23, 1),
        "mm345_nd_out": _bits(key, 24, 1),
        "has_dropout": _bits(key, 25, 1),
        "has_pse": _bits(key, 26, 1),
        "has_mask": _bits(key, 27, 1),
        "enable_l1_reuse": _bits(key, 28, 1),
        "tnd_s1_pingpong": _bits(key, 29, 1),
        "s1_template": _bits(key, 30, 4),
        "s2_template": _bits(key, 34, 4),
        "d_template": _bits(key, 38, 4),
        "deterministic": _bits(key, 42, 1),
        "has_rope": _bits(key, 43, 1),
    }


def _decode_fa_data(data: bytes) -> tuple[dict[str, int], dict[str, int]]:
    # FlashAttentionScoreGeneralTilingData (DAV_2201 generated header):
    # InputParams @ 0, MultiCoreParams @ 160, CoreParams @ 576.
    shape = {
        "B": _read(data, 0, "q"),
        "N2": _read(data, 8, "q"),
        "G": _read(data, 16, "q"),
        "S1": _read(data, 24, "q"),
        "S2": _read(data, 32, "q"),
        "D": _read(data, 48, "q"),
    }
    shape["N1"] = shape["N2"] * shape["G"]
    core = 576
    params = {
        "FA_CORE_NUM": _read(data, 160, "i"),
        "FA_S1_BASE": _read(data, core + 0, "i"),
        "FA_S1_TAIL": _read(data, core + 4, "i"),
        "FA_S1_OUTER": _read(data, core + 8, "q"),
        "FA_S1_VEC2_BASE": _read(data, core + 16, "i"),
        "FA_S1_VEC2_OUTER": _read(data, core + 24, "q"),
        "FA_S2_BASE": _read(data, core + 32, "i"),
        "FA_S2_TAIL": _read(data, core + 36, "i"),
        "FA_S2_OUTER": _read(data, core + 40, "q"),
        "FA_D_BASE": _read(data, core + 48, "i"),
        "FA_D_OUTER": _read(data, core + 56, "q"),
        "FA_N_RATIO": _read(data, core + 112, "i"),
    }
    return shape, params


def _decode_fag_common(data: bytes) -> dict[str, int]:
    return {
        "B": _read(data, 0, "q"),
        "N2": _read(data, 8, "q"),
        "G": _read(data, 16, "q"),
        "S1": _read(data, 24, "q"),
        "S2": _read(data, 32, "q"),
        "D": _read(data, 40, "q"),
        "ROPE_D": _read(data, 48, "q"),
        "DV": _read(data, 56, "q"),
        "CORE_NUM": _read(data, 104, "I"),
    }


def _decode_fag_generic(data: bytes) -> tuple[dict[str, int], dict[str, int]]:
    shape = _decode_fag_common(data)
    shape["N1"] = shape["N2"] * shape["G"]
    split = 184
    params = {
        "FAG_CORE_NUM": shape.pop("CORE_NUM"),
        "FAG_S1_OUTER": _read(data, split + 0, "q"),
        "FAG_S1_CV_RATIO": _read(data, split + 8, "I"),
        "FAG_S1_INNER": _read(data, split + 12, "I"),
        "FAG_S1_CV_INNER": _read(data, split + 16, "I"),
        "FAG_S2_OUTER": _read(data, split + 32, "q"),
        "FAG_S2_CV_RATIO": _read(data, split + 40, "I"),
        "FAG_S2_INNER": _read(data, split + 44, "I"),
        "FAG_BASE_MN": _read(data, split + 52, "I"),
        "FAG_BLOCK_OUTER": _read(data, split + 68, "I"),
    }
    return shape, params


def _decode_fag_same_ab(data: bytes) -> tuple[dict[str, int], dict[str, int]]:
    shape = _decode_fag_common(data)
    shape["N1"] = shape["N2"] * shape["G"]
    split = 192
    params = {
        "FAG_CORE_NUM": shape.pop("CORE_NUM"),
        "FAG_S1_OUTER": _read(data, split + 0, "q"),
        "FAG_S1_INNER": _read(data, split + 8, "I"),
        "FAG_S1_CV_INNER": _read(data, split + 12, "I"),
        "FAG_S2_OUTER": _read(data, split + 24, "q"),
        "FAG_S2_INNER": _read(data, split + 32, "I"),
        "FAG_S2_CV_INNER": _read(data, split + 36, "I"),
        "FAG_BASE_MN": _read(data, split + 44, "I"),
        "FAG_BLOCK_OUTER": _read(data, split + 48, "I"),
    }
    return shape, params


def _decode_fag_bn2(data: bytes) -> tuple[dict[str, int], dict[str, int]]:
    # FlashAttentionScoreGradTilingDataS1s2Bn2 (v8.5.0 arch32):
    # opInfo @ 0, splitCoreParams @ 232.
    shape = {
        "B": _read(data, 16, "q"),
        "N2": _read(data, 24, "q"),
        "S1": _read(data, 32, "q"),
        "S2": _read(data, 40, "q"),
        "G": _read(data, 48, "q"),
        "D": _read(data, 56, "q"),
    }
    shape["N1"] = shape["N2"] * shape["G"]
    shape["DV"] = shape["D"]
    split = 232
    params = {
        "FAG_CORE_NUM": _read(data, 0, "I"),
        "FAG_BASE_M": _read(data, split + 0, "I"),
        "FAG_BASE_N": _read(data, split + 4, "I"),
        "FAG_SINGLE_N": _read(data, split + 8, "I"),
        "FAG_SINGLE_M": _read(data, split + 12, "I"),
        "FAG_S1_OUTER": _read(data, split + 16, "I"),
        "FAG_S2_OUTER": _read(data, split + 20, "I"),
        "FAG_D_INNER": _read(data, split + 24, "I"),
        "FAG_SFT_BASE_M": _read(data, split + 28, "I"),
        "FAG_SFT_SINGLE_M": _read(data, split + 32, "I"),
    }
    return shape, params


def _decode_fag_mla(data: bytes) -> tuple[dict[str, int], dict[str, int]]:
    shape = {
        "B": _read(data, 8, "q"),
        "T1": _read(data, 16, "q"),
        "T2": _read(data, 24, "q"),
        "N2": _read(data, 32, "q"),
        "G": _read(data, 40, "q"),
        "D": _read(data, 48, "q"),
    }
    shape["N1"] = shape["N2"] * shape["G"]
    return shape, {"FAG_CORE_NUM": _read(data, 0, "I")}


def _fag_route(key_fields: dict[str, int]) -> str:
    axes = (key_fields["ub0_axis"], key_fields["ub1_axis"], key_fields["block_axis"])
    if axes == (0, 0, 0) and not any(key_fields.values()):
        return "fag_empty"
    if axes == (9, 9, 9):
        return "fag_basic_deterministic" if key_fields["deterministic"] else "fag_mla"
    if key_fields["same_ab"]:
        return "fag_same_ab"
    if axes == (4, 3, 4):
        return "fag_generic"
    if axes == (4, 3, 1):
        return "fag_s1s2_bn2_deterministic" if key_fields["deterministic"] else "fag_s1s2_bn2"
    if axes == (6, 6, 0):
        return "fag_b"
    if axes == (6, 6, 1):
        return "fag_n2"
    return "fag_axis_{}_{}_{}".format(*axes)


def parse_line(line: str) -> dict[str, object]:
    columns = line.rstrip("\n").split("\t")
    if not columns or columns[0] != PREFIX:
        raise ValueError("not an AUTOTILING_DUMP line")
    values = {}
    for column in columns[1:]:
        name, separator, value = column.partition("=")
        if not separator:
            raise ValueError(f"malformed dump column: {column}")
        values[name] = value

    operator = values["op"]
    key = int(values["key"])
    data = bytes.fromhex(values["data_hex"])
    workspace = [int(value) for value in values.get("workspace", "").split(",") if value]

    if operator == "FlashAttentionScore":
        key_fields = decode_fa_key(key)
        route = "fa_varlen" if key_fields["layout"] == 4 else "fa_general"
        shape, params = _decode_fa_data(data)
    elif operator == "FlashAttentionScoreGrad":
        key_fields = decode_fag_key(key)
        route = _fag_route(key_fields)
        if route == "fag_generic":
            shape, params = _decode_fag_generic(data)
        elif route == "fag_same_ab":
            shape, params = _decode_fag_same_ab(data)
        elif route in ("fag_s1s2_bn2", "fag_s1s2_bn2_deterministic"):
            shape, params = _decode_fag_bn2(data)
        elif route == "fag_mla":
            shape, params = _decode_fag_mla(data)
        else:
            shape, params = {}, {}
    else:
        raise ValueError(f"unsupported operator in dump: {operator}")

    return {
        "operator": operator,
        "case": values["case"],
        "route": route,
        "tiling_key": key,
        "key_fields": key_fields,
        "block_dim": int(values["block_dim"]),
        "workspace_sizes": workspace,
        "shape": shape,
        "tiling_params": params,
        "tiling_data_size": len(data),
    }


def parse_log(path: Path) -> list[dict[str, object]]:
    results = []
    seen = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if not line.startswith(PREFIX + "\t"):
            continue
        try:
            result = parse_line(line)
        except (KeyError, ValueError, struct.error) as error:
            print(f"{path}:{line_number}: {error}", file=sys.stderr)
            continue
        identity = (result["operator"], result["case"], result["tiling_key"], json.dumps(result["tiling_params"], sort_keys=True))
        if identity not in seen:
            seen.add(identity)
            results.append(result)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Decode official FA/FAG DAV_2201 host autotiling dumps")
    parser.add_argument("log", type=Path, help="transformer_op_host_ut output captured by get_tiling.sh")
    parser.add_argument("--output", type=Path, help="write all decoded records as JSON")
    args = parser.parse_args()

    results = parse_log(args.log)
    if args.output:
        args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for result in results:
        print(
            "{operator} {case}: route={route} key={tiling_key} block_dim={block_dim} "
            "shape={shape} tiling={tiling_params}".format(**result)
        )
    if not results:
        print("no AUTOTILING_DUMP records found", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
