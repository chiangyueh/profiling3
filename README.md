# FA/FAG tiling search

This repository adapts the generic optimization framework from the colleague's
Matmul example to CANN v8.5.0 `FlashAttentionScore` (FA) and
`FlashAttentionScoreGrad` (FAG).  Matmul operator sources are not part of this
repository.

## Execution flow

For one fixed shape, the PSO algorithm proposes several tiling candidates.  The
attention validator labels every candidate but does not block it.  Every
candidate is sent to the patched official Host tiling implementation and then
executed on the selected NPU:

```text
shape + PSO candidate
  -> FA/FAG validator label
  -> patched CANN Host tiling
  -> real FA/FAG tiling data
  -> compiled tiling-key kernel
  -> correctness and latency
```

For FA, the searched values are `FA_S1_BASE`, `FA_S2_BASE`, and
`FA_N_RATIO`.  The Host-side hook is defined in
`patches/ops_transformer_attention_search.patch`.  FAG hooks are defined in the
same patch for the generic and SameAB routes.

The tiling key selects a compiled kernel branch.  Candidate tiling values are
runtime Host-tiling inputs, so candidates sharing a key reuse the same compiled
binary.  A new key is compiled only on its first use and is then loaded from
`out/attention_cache`.

## Requirements

- CANN 8.5.0 and an Ascend NPU.
- The official `ops-transformer` v8.5.0 checkout at
  `../ops-transformer-official-8.5.0`, or set `OPS_TRANSFORMER_ROOT`.
- The checkout must be at commit
  `6ead121aded45355043b502756b6592fd7c30b14`.
- Python packages used by the colleague's framework: NumPy and pandas.

Compilation defaults to one job to avoid excessive host resource usage.  It
can be changed explicitly with `ATTENTION_BUILD_JOBS`.

## Run

Select a physical NPU explicitly:

```bash
python3 main.py --id=4
```

The active routes and shapes are configured in `ROUTE_CASES` inside
`attention_main.py`.  Each case stores its own tiling key, and the cases run
sequentially on the selected NPU.  The current configuration contains the two
FA routes that were first verified separately.

Each run writes:

- `results/attention_audit/validator_audit.jsonl`: one record per PSO candidate;
- `results/attention_audit/summary.json`: route-level completion and counts;
- `OPPROF_*`: temporary profiler output used to read latency.

The four audit categories are:

- `validator_accept_runtime_pass`;
- `validator_accept_runtime_fail`;
- `validator_reject_runtime_pass`;
- `validator_reject_runtime_fail`.

A failed candidate does not stop the remaining candidates.

## Relevant files

- `main.py`: the only user entry point;
- `attention_main.py`: shape, search domains, validator audit, and PSO wiring;
- `attention_bench.cpp`: FA/FAG ACLNN launcher and correctness output;
- `build_attention.sh`: single-key compilation and persistent cache;
- `run_attention.sh`: cached NPU execution and profiling;
- `tiling/`: generic search algorithms plus FA/FAG validators;
- `patches/ops_transformer_attention_search.patch`: candidate-to-Host-tiling bridge.

`get_tiling.sh` and `autotiling.py` are optional Host-only research utilities
for extracting official tiling packets.  They are not part of the normal NPU
search command.
