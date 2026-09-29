# FA/FAG search support

This directory contains the search framework copied from
`/usr/local/Ascend/gpt_research/information.zip` (`matmul_v3/tiling`).  The
GA, SA, PSO, cache and original estimators are retained.  The added code is a
thin operator adapter for the training operators:

- forward: `FlashAttentionScore`
- backward: `FlashAttentionScoreGrad`

`FusedInferAttentionScore` is intentionally not used as the forward operator.

## Layout

- `tiling/`: the imported search framework plus FA/FAG validators and estimators
- `attention_support.py`: JSON configuration and operator construction
- `search_attention.py`: common FA/FAG command-line entry point
- `configs/`: one initial search-space example for each operator
- `runners/`: adapters for real operator runners

## Check a configuration

This does not require an NPU or compile an operator:

```bash
python3 search_attention.py \
  --config configs/flash_attention_score.json \
  --check-only

python3 search_attention.py \
  --config configs/flash_attention_score_grad.json \
  --check-only
```

## Runner contract

The search framework exports every shape, constant, candidate and derived
parameter as an environment variable, then calls the selected runner with
`-r npu`, `-r sim`, or `-r npu --cycles-only`.

For forward, point `FA_FORWARD_RUNNER` at a `FlashAttentionScore` runner:

```bash
export FA_FORWARD_RUNNER=/absolute/path/to/forward_runner.sh
python3 search_attention.py --config configs/flash_attention_score.json
```

For backward, use `FA_BACKWARD_RUNNER`:

```bash
export FA_BACKWARD_RUNNER=/absolute/path/to/backward_runner.sh
python3 search_attention.py --config configs/flash_attention_score_grad.json
```

The downstream runner must do both of the following:

1. apply the exported candidate to the operator tiling or kernel launch;
2. fail on an incorrect result.

It may print `DURATION_US=<number>`.  If it does not, the estimator falls back
to the original `OPPROF_*` parsing.  A cycles runner prints `CYCLES=<number>`.

The official CANN host tilers do not automatically read the new environment
variables.  A production runner therefore needs a small source hook or an
equivalent direct-tiling launch.  Without that hook, different search
candidates would all execute the same official tiling and the result would not
be a real optimization.

## Candidate variables

Forward search uses:

- shape: `FA_B`, `FA_N1`, `FA_N2`, `FA_S1`, `FA_S2`, `FA_D`, `FA_DV`
- candidates: `FA_S1_BASE`, `FA_S2_BASE`, `FA_D_BASE`, `FA_CORE_NUM`
- derived: `FA_G`, `FA_S1_OUTER`, `FA_S2_OUTER`

Backward search uses:

- shape: `FAG_B`, `FAG_N1`, `FAG_N2`, `FAG_S1`, `FAG_S2`, `FAG_D`, `FAG_DV`
- candidates: `FAG_S1_INNER`, `FAG_S2_INNER`, `FAG_S1_CV_RATIO`,
  `FAG_S2_CV_RATIO`, `FAG_CORE_NUM`
- derived: `FAG_G`, `FAG_S1_OUTER`, `FAG_S2_OUTER`

These are deliberately route-neutral.  Route-specific restrictions should be
added to the JSON domains or to the downstream runner, rather than changing
GA/SA/PSO.
