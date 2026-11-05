from __future__ import annotations


def enabled(p):
    return bool(p["has_drop_mask"] and (p["S2"] % 8 or p["layout"] == "TND" and p["B"] > 1))
