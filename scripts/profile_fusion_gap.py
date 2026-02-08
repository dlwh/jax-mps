#!/usr/bin/env python3
"""Profile likely fusion/dispatch bottlenecks on jax-mps.

This script focuses on:
1) Matmul crossover size (where MPS starts outperforming CPU).
2) Elementwise throughput (add/tanh) across sizes.
3) Fused-vs-stepwise pointwise chains (proxy for fusion opportunity).
4) Tiny-op dispatch overhead (jit call + sync in a loop).
"""

from __future__ import annotations

import argparse
import statistics as stats
import time

import jax
import jax.numpy as jnp
import numpy as np


def sync_tree(x) -> None:
    jax.tree_util.tree_map(lambda y: y.block_until_ready(), x)


def bench_jit(fn, np_args, device, runs: int, warmup: int) -> tuple[float, float]:
    with jax.default_device(device):
        args = tuple(jax.device_put(a, device) for a in np_args)
        compiled = jax.jit(fn)
        sync_tree(compiled(*args))
        for _ in range(warmup):
            sync_tree(compiled(*args))
        timings = []
        for _ in range(runs):
            t0 = time.perf_counter()
            out = compiled(*args)
            sync_tree(out)
            timings.append((time.perf_counter() - t0) * 1000.0)
    return stats.median(timings), stats.pstdev(timings)


def chain20(x, y):
    z = x
    for i in range(20):
        z = jnp.tanh(z * 1.01 + y * 0.99)
        if i % 3 == 0:
            z = z + 0.1
        if i % 5 == 0:
            z = z * 0.9
    return z


def bench_chain_stepwise(np_args, device, runs: int, warmup: int) -> tuple[float, float]:
    with jax.default_device(device):
        x = jax.device_put(np_args[0], device)
        y = jax.device_put(np_args[1], device)
        mul = jax.jit(lambda a: a * 1.01)
        add = jax.jit(lambda a, b: a + b * 0.99)
        tanh = jax.jit(lambda a: jnp.tanh(a))

        z = x
        for _ in range(20):
            z = tanh(add(mul(z), y))
        z.block_until_ready()

        for _ in range(warmup):
            z = x
            for _ in range(20):
                z = tanh(add(mul(z), y))
            z.block_until_ready()

        timings = []
        for _ in range(runs):
            z = x
            t0 = time.perf_counter()
            for _ in range(20):
                z = tanh(add(mul(z), y))
            z.block_until_ready()
            timings.append((time.perf_counter() - t0) * 1000.0)
    return stats.median(timings), stats.pstdev(timings)


def bench_tiny_dispatch(device, size: int, n_calls: int) -> float:
    rng = np.random.default_rng(123)
    x_np = rng.standard_normal((size, size), dtype=np.float32)
    y_np = rng.standard_normal((size, size), dtype=np.float32)
    with jax.default_device(device):
        x = jax.device_put(x_np, device)
        y = jax.device_put(y_np, device)
        add = jax.jit(lambda a, b: a + b)
        add(x, y).block_until_ready()

        z = x
        t0 = time.perf_counter()
        for _ in range(n_calls):
            z = add(z, y)
            z.block_until_ready()
        dt = time.perf_counter() - t0
    return dt * 1e6 / n_calls


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--dispatch-calls", type=int, default=1000)
    parser.add_argument("--dispatch-size", type=int, default=128)
    args = parser.parse_args()

    cpu = jax.devices("cpu")[0]
    mps = jax.devices("mps")[0]
    rng = np.random.default_rng(0)

    print(f"Devices: cpu={cpu}, mps={mps}")

    print("\n[matmul_crossover] size,cpu_ms,mps_ms,cpu_over_mps")
    for size in [128, 256, 512, 768, 1024, 1536, 2048, 3072]:
        a = rng.standard_normal((size, size), dtype=np.float32)
        b = rng.standard_normal((size, size), dtype=np.float32)
        cpu_ms, _ = bench_jit(lambda x, y: x @ y, (a, b), cpu, args.runs, args.warmup)
        mps_ms, _ = bench_jit(lambda x, y: x @ y, (a, b), mps, args.runs, args.warmup)
        print(f"{size},{cpu_ms:.4f},{mps_ms:.4f},{cpu_ms / mps_ms:.3f}")

    print("\n[elementwise_add] size,cpu_ms,mps_ms,cpu_over_mps")
    for size in [128, 256, 512, 1024, 2048, 4096]:
        a = rng.standard_normal((size, size), dtype=np.float32)
        b = rng.standard_normal((size, size), dtype=np.float32)
        cpu_ms, _ = bench_jit(lambda x, y: x + y, (a, b), cpu, args.runs, args.warmup)
        mps_ms, _ = bench_jit(lambda x, y: x + y, (a, b), mps, args.runs, args.warmup)
        print(f"{size},{cpu_ms:.4f},{mps_ms:.4f},{cpu_ms / mps_ms:.3f}")

    print("\n[elementwise_tanh] size,cpu_ms,mps_ms,cpu_over_mps")
    for size in [128, 256, 512, 1024, 2048, 4096]:
        a = rng.standard_normal((size, size), dtype=np.float32)
        cpu_ms, _ = bench_jit(lambda x: jnp.tanh(x), (a,), cpu, args.runs, args.warmup)
        mps_ms, _ = bench_jit(lambda x: jnp.tanh(x), (a,), mps, args.runs, args.warmup)
        print(f"{size},{cpu_ms:.4f},{mps_ms:.4f},{cpu_ms / mps_ms:.3f}")

    chain_size = 1024
    a = rng.standard_normal((chain_size, chain_size), dtype=np.float32)
    b = rng.standard_normal((chain_size, chain_size), dtype=np.float32)
    cpu_fused_ms, _ = bench_jit(chain20, (a, b), cpu, args.runs, args.warmup)
    mps_fused_ms, _ = bench_jit(chain20, (a, b), mps, args.runs, args.warmup)
    cpu_step_ms, _ = bench_chain_stepwise((a, b), cpu, args.runs, args.warmup)
    mps_step_ms, _ = bench_chain_stepwise((a, b), mps, args.runs, args.warmup)
    print(
        "\n[chain20_fused_vs_stepwise] "
        f"cpu_fused_ms={cpu_fused_ms:.4f} cpu_stepwise_ms={cpu_step_ms:.4f} "
        f"mps_fused_ms={mps_fused_ms:.4f} mps_stepwise_ms={mps_step_ms:.4f}"
    )

    cpu_dispatch = bench_tiny_dispatch(cpu, args.dispatch_size, args.dispatch_calls)
    mps_dispatch = bench_tiny_dispatch(mps, args.dispatch_size, args.dispatch_calls)
    print(
        "\n[tiny_dispatch_us_per_call] "
        f"size={args.dispatch_size} calls={args.dispatch_calls} "
        f"cpu={cpu_dispatch:.2f} mps={mps_dispatch:.2f} "
        f"cpu_over_mps={cpu_dispatch / mps_dispatch:.3f}"
    )


if __name__ == "__main__":
    main()
