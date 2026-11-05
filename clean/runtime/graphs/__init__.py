import copy
from collections import Counter

import numpy as np

from ..features import TRANSFER_PARAMETERS
from ..native.graph import compact_graph
from .engine import Graph


def build(work):
    components, seen = [], set()
    for pair in work.pairs:
        identity = tuple(tuple(id(job) for job in tasks) for tasks in pair)
        if identity in seen:
            continue
        seen.add(identity)
        component = copy.copy(work)
        component.pairs = [pair]
        components.append(compact_graph(Graph(component).build()))
    if len(components) == 1:
        return components[0]
    a, b, labels, templates, phases, roots = [], [], [], [], [], []
    offset, template_offset, events = 0, 0, Counter()
    for graph in components:
        a.append(np.where(graph['a'] >= 0, graph['a'] + offset, -1).astype(np.int32))
        b.append(np.where(graph['b'] >= 0, graph['b'] + offset, -1).astype(np.int32))
        labels.append(graph['which'] + template_offset)
        templates.append(graph['templates'])
        phases.extend(graph['phases'])
        roots.append(graph['root'] + offset)
        offset += len(graph['a'])
        template_offset += len(graph['templates'])
        events.update(graph['execution_events'])
    if work.manual_transfer:
        zero = template_offset
        templates.append(np.zeros((1, len(TRANSFER_PARAMETERS))))
        phases.append("startup")
        left, right, which = (
            list(np.concatenate(a)),
            list(np.concatenate(b)),
            list(np.concatenate(labels)),
        )
        while len(roots) > 1:
            first, second = roots.pop(), roots.pop()
            roots.append(len(left))
            left.append(first)
            right.append(second)
            which.append(zero)
        return {
            "a": np.ascontiguousarray(left, dtype=np.int32),
            "b": np.ascontiguousarray(right, dtype=np.int32),
            "which": np.ascontiguousarray(which, dtype=np.int32),
            "templates": np.ascontiguousarray(np.vstack(templates)),
            "root": roots[0],
            "phases": phases,
            "execution_events": dict(events),
        }
    root_a, root_b = [], []
    root = roots[0]
    for value in roots[1:]:
        root_a.append(root)
        root_b.append(value)
        root = offset + len(root_a) - 1
    return {'a': np.concatenate(a + [np.array(root_a, dtype=np.int32)]),
            'b': np.concatenate(b + [np.array(root_b, dtype=np.int32)]),
            'which': np.concatenate(labels + [np.zeros(len(root_a), dtype=np.int32)]),
            'templates': np.vstack(templates), 'phases': phases, 'root': root,
            'execution_events': dict(events)}
