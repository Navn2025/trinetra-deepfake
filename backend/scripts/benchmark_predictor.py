"""
Measures real inference latency for predictor.predict_fake_probability on
this machine's actual hardware/checkpoint -- not an estimate. Run with:
  venv\\Scripts\\python.exe scripts\\benchmark_predictor.py
"""
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image

import predictor

N_WARMUP = 5
N_RUNS = 100


def main():
    image = Image.new("RGB", (256, 256), (110, 130, 150))

    load_start = time.perf_counter()
    predictor._load_models()
    load_elapsed = time.perf_counter() - load_start
    print(f"Model load time: {load_elapsed * 1000:.1f} ms  (device={predictor.DEVICE})")

    for _ in range(N_WARMUP):
        predictor.predict_fake_probability(image)
    if predictor.DEVICE.type == "cuda":
        torch.cuda.synchronize()

    latencies_ms = []
    for _ in range(N_RUNS):
        start = time.perf_counter()
        predictor.predict_fake_probability(image)
        if predictor.DEVICE.type == "cuda":
            torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - start) * 1000)

    latencies_ms.sort()

    def pct(p):
        idx = min(int(len(latencies_ms) * p) , len(latencies_ms) - 1)
        return latencies_ms[idx]

    print(f"\npredict_fake_probability() over {N_RUNS} warm calls, single 256x256 crop:")
    print(f"  mean: {statistics.mean(latencies_ms):.2f} ms")
    print(f"  p50:  {pct(0.50):.2f} ms")
    print(f"  p95:  {pct(0.95):.2f} ms")
    print(f"  p99:  {pct(0.99):.2f} ms")
    print(f"  min:  {min(latencies_ms):.2f} ms")
    print(f"  max:  {max(latencies_ms):.2f} ms")
    print(f"  throughput (single-worker, sequential): {1000 / statistics.mean(latencies_ms):.1f} req/s")

    if predictor.DEVICE.type == "cuda":
        print(f"\nGPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print(f"GPU memory reserved:  {torch.cuda.memory_reserved() / 1024**2:.1f} MB")


if __name__ == "__main__":
    main()
