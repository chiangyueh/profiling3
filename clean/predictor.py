if __package__:
    from .runtime import costs, graphs, timing, workloads
else:
    from runtime import costs, graphs, timing, workloads


def predict(inputs):
    model = costs.load_model()
    work, implementation = workloads.build(inputs, model["hardware"], model["api_input_normalization"])
    graph = graphs.build(work)
    value, _, phases = timing.evaluate(graph, costs.rates(work, graph, model))
    return {"predicted_us": float(value), "implementation": implementation,
            "critical_path_phase_us": phases}
