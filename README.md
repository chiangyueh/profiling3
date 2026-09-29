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

## Read official autotiling

The MatMul `dict.h` method from `ops-nn.zip` is connected to FA/FAG through the
operators' real host unit-test path.  The shared test executor emits the tiling
key, block dimension, workspaces, and exact raw tiling bytes only when
`AUTOTILING_DUMP=1`; normal unit-test output is unchanged.  The decoder then
maps the key to a route and reads the route's named tiling fields.

Put the DeepSeek/Pangu cases into the official host tests (forward uses its
CSV; backward may use the same generated-`dict.h` pattern as MatMul), then run:

```bash
./get_tiling.sh forward
./get_tiling.sh backward
# or both:
./get_tiling.sh all
```

`OPS_TRANSFORMER_ROOT` defaults to `../ops-transformer`.  The command produces
`autotiling_run.log` and `autotiling_results.json`.  Use
`AUTOTILING_FILTER='exact.gtest.filter'` to collect only generated model cases.
If the opt-in dump hook is absent, the script first verifies and applies the
included source patch.  The build needs a complete official checkout and CMake
3.18.4 or newer.

The decoded record is the official baseline, not a benchmark winner.  Its
`route` selects the validator; its `tiling_params` show the actual starting
packet and which fields can be searched without changing kernel family.

## Run

1. Run the official host autotiling collector for the real DeepSeek/Pangu
   cases and identify each route in `autotiling_results.json`.
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
