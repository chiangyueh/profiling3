import os
from pathlib import Path


def cann_root():
    choices = [
        os.environ.get(name, "")
        for name in (
            "ASCEND_CANN_PACKAGE_PATH",
            "ASCEND_HOME_PATH",
            "ASCEND_TOOLKIT_HOME",
        )
    ]
    choices += [
        "/usr/local/Ascend/cann-8.5.0",
        "/usr/local/Ascend/ascend-toolkit/latest",
        "/usr/local/Ascend/cann-8.5.2",
    ]
    for raw in choices:
        p = Path(raw)
        headers = any(
            (p / prefix / "tiling/matmul/matmul_tiling.h").is_file()
            for prefix in ("include", "include/ascendc/host_api")
        )
        library = any(
            (p / "lib64" / name).is_file()
            for name in ("libtiling_api.so", "libtiling_api.a")
        )
        if raw and headers and library:
            return p.resolve()
    raise RuntimeError("CANN Host Matmul tiling API headers/library are missing")


def cann_environment(sdk):
    environment = os.environ.copy()
    paths = [str(sdk / "lib64")]
    paths.extend(
        path
        for path in environment.get("LD_LIBRARY_PATH", "").split(os.pathsep)
        if path and "devlib" not in Path(path).parts
    )
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(dict.fromkeys(paths))
    return environment
