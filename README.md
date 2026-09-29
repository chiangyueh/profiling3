# FA/FAG search framework

The search framework comes from `information.zip/matmul_v3/tiling` and keeps
the original execution flow:

```text
main.py -> BaseAlgo -> run.sh
```

There is no separate CLI, JSON configuration, check-only mode, or mock runner.

## Usage

Edit `OPERATOR`, `SIZES`, and the corresponding domains in `main.py`, exactly
as the MatMul version edits `SIZES` and `get_domains()`. Then run:

```bash
python3 main.py
```

Supported values of `OPERATOR` are:

```python
OPERATOR = "flash_attention_score"
OPERATOR = "flash_attention_score_grad"
```

The optimizer continues to invoke:

```bash
./run.sh -r npu
```

## Current runner boundary

The repository does not yet contain a real FA/FAG executable. Until the
official operator runner is integrated, set the matching runner path:

```bash
export FA_FORWARD_RUNNER=/absolute/path/to/real_forward_runner.sh
# or
export FA_BACKWARD_RUNNER=/absolute/path/to/real_backward_runner.sh
```

The real runner must apply the exported candidate tiling parameters and reject
incorrect output. Merely launching the default official tiling would not be a
valid optimization measurement.
