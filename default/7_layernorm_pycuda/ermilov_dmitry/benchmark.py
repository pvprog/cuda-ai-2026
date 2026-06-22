import time
import numpy as np

from layernorm_pycuda import layernorm_pycuda


def layernorm_ref(input, gamma, beta, row_size, eps=1e-5):
    x = np.asarray(input, dtype=np.float32).reshape(-1, row_size)
    gamma = np.asarray(gamma, dtype=np.float32)
    beta = np.asarray(beta, dtype=np.float32)

    mean = np.mean(x, axis=1, keepdims=True)
    var = np.var(x, axis=1, keepdims=True)

    y = (x - mean) / np.sqrt(var + eps)
    y = y * gamma + beta

    return y.reshape(-1)


def benchmark(func, iterations=5):
    best = float("inf")

    for _ in range(iterations):
        start = time.perf_counter()

        result = func()

        elapsed = time.perf_counter() - start
        best = min(best, elapsed)

        _ = float(result[len(result) // 2])

    return best


def main():
    ROW_COUNT = 4096
    ROW_SIZE = 1024

    rng = np.random.default_rng(42)

    x = rng.uniform(-5.0, 5.0,
                    size=ROW_COUNT * ROW_SIZE).astype(np.float32)

    gamma = rng.uniform(0.5, 1.5,
                        size=ROW_SIZE).astype(np.float32)

    beta = rng.uniform(-1.0, 1.0,
                       size=ROW_SIZE).astype(np.float32)

    print(f"Matrix size: {ROW_COUNT} x {ROW_SIZE}\n")

    ref_time = benchmark(
        lambda: layernorm_ref(x, gamma, beta, ROW_SIZE)
    )

    cuda_time = benchmark(
        lambda: layernorm_pycuda(x, gamma, beta, ROW_SIZE)
    )

    ref = layernorm_ref(x, gamma, beta, ROW_SIZE)
    gpu = layernorm_pycuda(x, gamma, beta, ROW_SIZE)

    max_error = np.max(np.abs(ref - gpu))

    print(f"Reference : {ref_time:.6f} s")
    print(f"PyCUDA    : {cuda_time:.6f} s")
    print(f"Speedup   : {ref_time / cuda_time:.2f}x")
    print(f"Max error : {max_error:.6e}")


if __name__ == "__main__":
    main()