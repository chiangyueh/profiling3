import numpy as np

from .features import PHASES
from .native.graph import graph_time


def evaluate(graph, rates):
    durations = graph["templates"] @ np.asarray(rates)
    extra = np.zeros((len(durations), len(PHASES)))
    for index, phase in enumerate(graph["phases"]):
        extra[index, PHASES.index(phase)] = durations[index]
    encoded = {**graph, "templates": np.column_stack((graph["templates"], extra))}
    value, derivative = graph_time(
        encoded, np.append(rates, np.zeros(len(PHASES))), True
    )
    result = dict(zip(PHASES, map(float, derivative[-len(PHASES) :])))
    return value, derivative[: len(rates)], result
