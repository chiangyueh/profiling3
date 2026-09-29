from __future__ import annotations

from pathlib import Path
import json

from tiling.algs.ga import GaAttentionValidator
from tiling.base import BaseParam
from tiling.limits import AttentionLimits
from tiling.validators.AttentionValidators import (
    BACKWARD,
    FORWARD,
    SUPPORTED_OPERATORS,
    AttentionValidator,
)


SHAPE_FIELDS = ("B", "N1", "N2", "S1", "S2", "D", "DV")
REQUIRED_DOMAINS = {
    FORWARD: ("FA_S1_BASE", "FA_S2_BASE", "FA_D_BASE", "FA_CORE_NUM"),
    BACKWARD: (
        "FAG_S1_INNER",
        "FAG_S2_INNER",
        "FAG_S1_CV_RATIO",
        "FAG_S2_CV_RATIO",
        "FAG_CORE_NUM",
    ),
}


def load_attention_config(path: str | Path) -> dict:
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    validate_attention_config(config)
    configured_runner = config.get("runner")
    if configured_runner:
        runner_path = Path(configured_runner)
        if not runner_path.is_absolute():
            runner_path = config_path.parent / runner_path
        config["runner"] = str(runner_path.resolve())
    config["_config_path"] = str(config_path)
    return config


def validate_attention_config(config: dict) -> None:
    operator = config.get("operator")
    if operator not in SUPPORTED_OPERATORS:
        raise ValueError(f"operator must be one of {SUPPORTED_OPERATORS}")

    shape = config.get("shape")
    if not isinstance(shape, dict):
        raise ValueError("shape must be an object")
    missing_shape = [name for name in SHAPE_FIELDS if name not in shape]
    if missing_shape:
        raise ValueError(f"missing shape fields: {missing_shape}")
    if any(not isinstance(shape[name], int) or shape[name] <= 0 for name in SHAPE_FIELDS):
        raise ValueError("all shape fields must be positive integers")
    if shape["N1"] % shape["N2"] != 0:
        raise ValueError("N1 must be divisible by N2")

    domains = config.get("domains")
    if not isinstance(domains, dict):
        raise ValueError("domains must be an object")
    missing_domains = [name for name in REQUIRED_DOMAINS[operator] if name not in domains]
    if missing_domains:
        raise ValueError(f"missing search domains: {missing_domains}")
    for name, domain in domains.items():
        if not isinstance(domain, list) or not domain:
            raise ValueError(f"domain {name} must be a non-empty list")
        if any(not isinstance(value, int) or value <= 0 for value in domain):
            raise ValueError(f"domain {name} must contain positive integers")
        if len(domain) != len(set(domain)):
            raise ValueError(f"domain {name} contains duplicates")

    max_cores = config.get("max_cores", 24)
    if not isinstance(max_cores, int) or max_cores <= 0:
        raise ValueError("max_cores must be a positive integer")

    bounds = (
        {
            "FA_S1_BASE": shape["S1"],
            "FA_S2_BASE": shape["S2"],
            "FA_D_BASE": max(shape["D"], shape["DV"]),
            "FA_CORE_NUM": max_cores,
        }
        if operator == FORWARD
        else {
            "FAG_S1_INNER": shape["S1"],
            "FAG_S2_INNER": shape["S2"],
            "FAG_CORE_NUM": max_cores,
        }
    )
    for name, upper in bounds.items():
        if not any(value <= upper for value in domains[name]):
            raise ValueError(f"domain {name} has no value at or below {upper}")

    constants = config.get("constants", {})
    if not isinstance(constants, dict):
        raise ValueError("constants must be an object")
    if any(not isinstance(value, int) for value in constants.values()):
        raise ValueError("runner constants must be integers")


def make_input_params(config: dict) -> list[BaseParam]:
    prefix = "FA" if config["operator"] == FORWARD else "FAG"
    values = {f"{prefix}_{name}": value for name, value in config["shape"].items()}
    for name, value in config.get("constants", {}).items():
        if name in values:
            raise ValueError(f"constant duplicates a shape parameter: {name}")
        values[name] = value
    return [BaseParam(name=name, value=value, is_const=True) for name, value in values.items()]


def make_validator(config: dict, use_ga_params: bool = False) -> AttentionValidator:
    limits = AttentionLimits(
        domains=config["domains"],
        operator=config["operator"],
        max_cores=config.get("max_cores", 24),
    )
    validator_type = GaAttentionValidator if use_ga_params else AttentionValidator
    return validator_type(limits)
