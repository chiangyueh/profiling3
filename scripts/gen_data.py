

import numpy as np
import os


def gen_golden_data(m: int, n: int, k: int):
    print(f"[gen_data] M={m} N={n} K={k}")

    x1_gm = np.random.randint(1, 10, [m, k]).astype(np.float16)
    x2_gm = np.random.randint(1, 10, [k, n]).astype(np.float16)
    golden = np.matmul(x1_gm.astype(np.float32), x2_gm.astype(np.float32)).astype(np.float32)
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    x1_gm.tofile("./input/x1_gm.bin")
    x2_gm.tofile("./input/x2_gm.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    M = int(os.environ.get("MM_M", 512))
    N = int(os.environ.get("MM_N", 512))
    K = int(os.environ.get("MM_K", 512))
    gen_golden_data(M, N, K)