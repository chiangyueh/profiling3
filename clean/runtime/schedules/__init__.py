from .. import inputs as cost
from . import b_schedule, drop_adapter_schedule, general_schedule, s1_schedule, sameab_schedule, varlen_schedule


def select(inputs, hardware):
    p = cost.normalize(inputs)
    if p["dtype"] not in ("FP16", "BF16") or p["layout"] not in ("BNSD", "SBH", "BSND", "TND"):
        raise ValueError("source model supports dense FP16/BF16 BNSD/SBH/BSND and uniform-length TND")
    if int(hardware["aiv_cores"]) != 2 * int(hardware["aic_cores"]):
        raise ValueError("source model requires two Vector cores per Cube")
    try:
        s = sameab_schedule.schedule(p, hardware)
    except ValueError as error:
        if str(error) != "inputs do not select a supported SameAB kernel":
            raise
        branch = (
            varlen_schedule if p["layout"] == "TND" else
            general_schedule if p["S2"] > 1024 else
            b_schedule if b_schedule.is_capable(p, hardware) else s1_schedule
        )
        s = branch.schedule(p, hardware)
    s["drop_adapter"] = drop_adapter_schedule.enabled(p)
    if p["layout"] == "TND" and not s["same_ab"]:
        s["l1"] = s["s1_base"] <= 128 and max(p["D"], p["DV"]) <= 256
    return s


def module(schedule):
    if schedule["same_ab"]:
        return sameab_schedule
    return {
        "b": b_schedule,
        "b_drop": b_schedule,
        "s1": s1_schedule,
        "general": general_schedule,
        "varlen": varlen_schedule,
    }[schedule["family"]]


def implementation(p, s):
    if p["layout"] == "TND":
        return (
            "varlen_same_ab"
            if s["same_ab"]
            else "varlen_l1" if s["l1"] else "varlen_gm"
        )
    return "b" if s["family"] in ("b", "b_drop") else s["family"]


def cost_class(p, s):
    family = implementation(p, s)
    path = "paired_reuse" if s["reuse"] else "l1_carry" if s["l1"] else "gm"
    return "|".join(
        (
            family,
            p["dtype"],
            p["layout"],
            path,
            "NZ" if s["nz"] else "ND",
            "STATIC" if s.get("static_matmul") else s["policy"],
        )
    )


def configure_matmul(p, s, mm):
    a, b = mm
    static = (
        p["dtype"] == "BF16"
        and not p["has_drop_mask"]
        and p["layout"] in ("BSND", "SBH")
        and s["family"] in ("b", "s1")
    )
    static = static and (a["base_m"], a["base_n"], a["base_k"]) == (
        b["base_m"],
        b["base_k"],
        b["base_n"],
    )
    templates = (
        (16, 16, 80),
        (32, 32, 80),
        (48, 48, 80),
        (64, 64, 80),
        (16, 16, 96),
        (32, 32, 96),
    )
    static = static and (
        (a["base_m"], a["base_n"], a["base_k"]) in templates
        if s["family"] == "b"
        else a["base_m"] == a["base_n"] == 128 and a["base_k"] in (80, 96, 128)
    )
    s["static_matmul"] = bool(
        static and (p["layout"] == "BSND" or a["base_k"] == 96)
    )
