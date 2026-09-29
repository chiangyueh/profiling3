# FA/FAG tiling search

This keeps the workflow of `kernel-operator-tiling.zip`: the search algorithms
are shared, while each kernel route owns its validator.

```text
main.py -> common search algorithm -> route validator -> run.sh
```

The validators were derived from the DAV_2201 host tiling in
`ops-transformer/attention/flash_attention_score` and
`flash_attention_score_grad`. They validate a selected route; they do not guess
the route from latency or switch kernels during a search.

## Run

1. Profile the real DeepSeek/Pangu workload and identify its host tiling route.
2. In `main.py`, set `KERNEL`, `SIZES`, `FEATURES`, and that route's domains.
3. Point the wrapper at the real evaluator and run the same single entry point:

```bash
export FA_FORWARD_RUNNER=/absolute/path/to/forward_runner.sh
# or, for FAG:
export FA_BACKWARD_RUNNER=/absolute/path/to/backward_runner.sh
python3 main.py
```

Supported route validators are:

| `KERNEL` | Official DAV_2201 host route | Priority |
|---|---|---:|
| `fa_general` | S1S2 BN2GS1 | 96 |
| `fa_varlen` | TND VarLen | 94 |
| `fag_mla` | MLA Basic | 1001 |
| `fag_same_ab` | SameAB / SameAB deterministic | 15500 / 1100 |
| `fag_generic` | Generic S1S2 | 16000 |

`fag_mla` has no searchable host tile; its empty domain represents the one
fixed legal configuration. `fag_generic` also recognizes the guarded S2
full-wave anchor implemented in the local `ops-transformer` branch.

`run.sh` receives every shape, feature, derived field, and candidate tiling
parameter as an environment variable. The real evaluator must apply those
parameters, write `output/output.bin` and `output/golden.bin`, and emit the
normal `OPPROF_*/OpBasicInfo.csv`. A failed evaluator or a missing/stale output
is treated as an invalid candidate.

The uploaded ZIP contains MatMul examples but no DeepSeek/Pangu FA workload
list. Do not treat the example shape in `main.py` as a model workload; replace
it with the profiled shapes before running a campaign.
