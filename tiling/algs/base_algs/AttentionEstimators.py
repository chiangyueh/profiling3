from tiling.algs.base_algs.CycleEstimator import BaseAlgoNPUCycles
from tiling.algs.base_algs.MsprofEstimators import BaseAlgoMsprof, BaseAlgoProfile


class _AttentionCorrectness:
    """The real FA/FAG runner owns correctness verification."""

    def _is_right(self, *args, **kwargs) -> bool:
        return True


class AttentionAlgoProfile(_AttentionCorrectness, BaseAlgoProfile):
    pass


class AttentionAlgoMsprof(_AttentionCorrectness, BaseAlgoMsprof):
    pass


class AttentionAlgoNPUCycles(_AttentionCorrectness, BaseAlgoNPUCycles):
    pass
