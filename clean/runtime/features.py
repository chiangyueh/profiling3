PIPES = {
    "cube": ("aic_cube_active_sum_us", ("flops_m", "tiles", "output_tiles")),
    "mte1": ("aic_mte1_active_sum_us", ("l0_kb", "tiles")),
    "mte2": ("aic_mte2_active_sum_us", ("read_kb", "loads", "address_gap_kb")),
    "fixpipe": (
        "aic_fixpipe_active_sum_us",
        ("nd_write_kb", "nz_write_kb", "stores", "calls"),
    ),
    "vector": (
        "aiv_vec_active_sum_us",
        (
            "softmax_k",
            "reduction_k",
            "cast_k",
            "nz2nd_k",
            "nd2nz_k",
            "online_k",
            "rescale_k",
            "normalize_k",
            "drop_k",
            "unpack_k",
            "chunks",
            "scale_k",
            "duplicate_k",
        ),
    ),
    "vmte2": ("aiv_mte2_active_sum_us", ("read_kb", "alignment_read_kb", "loads")),
    "vmte3": ("aiv_mte3_active_sum_us", ("write_kb", "stores", "address_gap_kb")),
    "cscalar": ("aic_scalar_active_sum_us", ("workers", "batch_entries", "commands")),
    "vscalar": (
        "aiv_scalar_active_sum_us",
        ("workers", "batch_entries", "chunks", "drop_chunks", "adapter_workers"),
    ),
}
PHASES = ("startup", "drop_adapter", "bmm1", "vec1", "bmm2", "vec2")
PARAMETERS = tuple((pipe, name) for pipe, (_, names) in PIPES.items() for name in names)
TRANSFER_FEATURES = ("payload_kb", "copy_calls", "strided_rows", "nd2nz_partial_rows", "nd2nz_split_sector_kb", "capacity_pressure_kb")
TRANSFER_PARAMETERS = PARAMETERS + tuple(("mte2", name) for name in TRANSFER_FEATURES)
