#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from attention_support import load_attention_config, make_input_params, make_validator
from tiling.algs.base_algs import AttentionAlgoMsprof, AttentionAlgoNPUCycles, AttentionAlgoProfile
from tiling.algs.ga import GaAlgo
from tiling.algs.pso import PsoAlgo
from tiling.algs.sa import SaAlgo


SEARCH_ALGORITHMS = {"ga": GaAlgo, "pso": PsoAlgo, "sa": SaAlgo}
ESTIMATORS = {
    "npu": AttentionAlgoProfile,
    "sim": AttentionAlgoMsprof,
    "cycles": AttentionAlgoNPUCycles,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search FA/FAG tiling candidates")
    parser.add_argument("--config", required=True, help="FA or FAG JSON configuration")
    parser.add_argument("--runner", help="override the runner from the configuration")
    parser.add_argument("--algorithm", choices=SEARCH_ALGORITHMS)
    parser.add_argument("--mode", choices=ESTIMATORS)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--cache")
    parser.add_argument("--output")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def _algorithm_type(algorithm: str, mode: str) -> type:
    return type(
        f"Attention{algorithm.title()}{mode.title()}",
        (ESTIMATORS[mode], SEARCH_ALGORITHMS[algorithm]),
        {},
    )


def _result_dict(result) -> dict:
    return {
        "duration": result.duration,
        "params": {param.name: param.value for param in result.params},
    }


def main() -> int:
    args = _parser().parse_args()
    config = load_attention_config(args.config)
    algorithm = args.algorithm or config.get("algorithm", "pso")
    mode = args.mode or config.get("mode", "npu")
    steps = args.steps or config.get("steps", 32)
    if steps <= 0:
        raise ValueError("steps must be positive")

    use_ga_params = algorithm == "ga"
    validator = make_validator(config, use_ga_params=use_ga_params)
    input_params = make_input_params(config)
    if args.check_only:
        sample = validator.get_combinations(1, input_params)[0]
        print(json.dumps({
            "operator": config["operator"],
            "algorithm": algorithm,
            "mode": mode,
            "sample": {param.name: param.value for param in sample},
        }, indent=2, sort_keys=True))
        return 0

    runner = args.runner or config.get("runner")
    if not runner:
        raise ValueError("a runner is required unless --check-only is used")
    runner_path = Path(runner).resolve()
    if not runner_path.is_file():
        raise ValueError(f"runner does not exist: {runner_path}")

    cache = args.cache or config.get("cache", f"cache_{config['operator']}_{mode}.json")
    common = {
        "is_stop": lambda results: len(results) >= steps,
        "validator": validator,
        "input_params": input_params,
        "runner": str(runner_path),
        "cache_path": cache,
        "verbose": args.verbose,
    }
    if algorithm == "ga":
        common["pop_size"] = config.get("population", 8)
    elif algorithm == "pso":
        common["swarm_size"] = config.get("population", 8)

    results = _algorithm_type(algorithm, mode)(**common)()
    best = min(results, key=lambda result: result.duration)
    payload = {
        "operator": config["operator"],
        "algorithm": algorithm,
        "mode": mode,
        "best": _result_dict(best),
        "steps": [_result_dict(result) for result in results],
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
