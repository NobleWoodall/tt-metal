#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
ResNet50 performance benchmark on a single Blackhole TT device.

Usage:
    cd /home/ttuser/noble_workspace/tt-metal
    source python_env/bin/activate
    export ARCH_NAME=blackhole
    export TT_METAL_HOME=$(pwd)
    export PYTHONPATH=$(pwd)
    export TT_METAL_DISABLE_FABRIC_TWO_ERISC=1

    python benchmark_resnet.py
    python benchmark_resnet.py --runs 20 --output results/resnet50
    python benchmark_resnet.py --batch-sizes 16 32 --fidelities LoFi HiFi2
"""

import argparse
import csv
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchvision

import ttnn
from models.demos.ttnn_resnet.tests.common.resnet50_test_infra import create_test_infra


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FIDELITY_MAP = {
    "LoFi": ttnn.MathFidelity.LoFi,
    "HiFi2": ttnn.MathFidelity.HiFi2,
    "HiFi4": ttnn.MathFidelity.HiFi4,
}

DEFAULT_BATCH_SIZES = [16, 32]
DEFAULT_FIDELITIES = ["LoFi", "HiFi2"]
DEFAULT_RUNS = 10  # inference runs after warmup
DEFAULT_WARMUP_RUNS = 2  # runs to discard before measuring
DEFAULT_OUTPUT_DIR = "generated/benchmarks/resnet50"


# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    model: str
    batch_size: int
    math_fidelity: str
    weight_dtype: str
    act_dtype: str
    num_devices: int
    # timing (all in seconds unless noted)
    init_time_s: float = 0.0  # model init + weight transfer
    compile_time_s: float = 0.0  # first inference (op compilation)
    warmup_time_s: float = 0.0  # discarded warm-up runs (total)
    mean_latency_ms: float = 0.0
    std_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    throughput_img_s: float = 0.0  # samples / second


# ---------------------------------------------------------------------------
# Benchmark one configuration
# ---------------------------------------------------------------------------


def run_one(
    device,
    batch_size: int,
    fidelity_name: str,
    weight_dtype=ttnn.bfloat8_b,
    act_dtype=ttnn.bfloat8_b,
    num_runs: int = DEFAULT_RUNS,
    num_warmup: int = DEFAULT_WARMUP_RUNS,
) -> BenchmarkResult:
    num_devices = device.get_num_devices()
    fidelity = FIDELITY_MAP[fidelity_name]
    total_batch = batch_size * num_devices

    result = BenchmarkResult(
        model="resnet50",
        batch_size=total_batch,
        math_fidelity=fidelity_name,
        weight_dtype=str(weight_dtype),
        act_dtype=str(act_dtype),
        num_devices=num_devices,
    )

    # ---- init -------------------------------------------------------
    t0 = time.perf_counter()
    test_infra = create_test_infra(
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        fidelity,
        dealloc_input=True,
        final_output_mem_config=ttnn.L1_MEMORY_CONFIG,
        model_location_generator=None,
    )
    tt_inputs_host, input_mem_config = test_infra.setup_l1_sharded_input(device)
    ttnn.synchronize_device(device)
    result.init_time_s = time.perf_counter() - t0

    def single_run():
        test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
        out = test_infra.run()
        ttnn.synchronize_device(device)
        _ = ttnn.to_torch(out, mesh_composer=test_infra.output_mesh_composer)
        out.deallocate()

    # ---- compile (first run) ----------------------------------------
    t0 = time.perf_counter()
    single_run()
    result.compile_time_s = time.perf_counter() - t0

    # ---- warmup (discarded) -----------------------------------------
    t0 = time.perf_counter()
    for _ in range(num_warmup):
        single_run()
    result.warmup_time_s = time.perf_counter() - t0

    # ---- measurement ------------------------------------------------
    latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        single_run()
        latencies.append((time.perf_counter() - t0) * 1e3)  # ms

    import statistics

    result.mean_latency_ms = statistics.mean(latencies)
    result.std_latency_ms = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
    result.min_latency_ms = min(latencies)
    result.max_latency_ms = max(latencies)
    result.throughput_img_s = total_batch / (result.mean_latency_ms / 1e3)

    return result


# ---------------------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------------------


def save_csv(results: list[BenchmarkResult], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=asdict(results[0]).keys())
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
    print(f"CSV saved → {path}")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def save_plots(results: list[BenchmarkResult], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = [f"bs{r.batch_size}\n{r.math_fidelity}" for r in results]
    latencies = [r.mean_latency_ms for r in results]
    stds = [r.std_latency_ms for r in results]
    throughputs = [r.throughput_img_s for r in results]

    x = range(len(results))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("ResNet50 – Blackhole Benchmark", fontsize=13)

    # Latency
    bars = ax1.bar(x, latencies, yerr=stds, capsize=4, color="steelblue", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("End-to-end latency (ms)")
    ax1.set_title("Per-batch Latency")
    for bar, val in zip(bars, latencies):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Throughput
    bars2 = ax2.bar(x, throughputs, color="darkorange", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("Throughput (images / s)")
    ax2.set_title("Inference Throughput")
    for bar, val in zip(bars2, throughputs):
        ax2.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 10, f"{val:.0f}", ha="center", va="bottom", fontsize=8
        )

    plt.tight_layout()
    plot_path = out_dir / "resnet50_benchmark.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plot saved → {plot_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="ResNet50 benchmark on Blackhole")
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_BATCH_SIZES,
        help="Per-device batch sizes (supported: 16, 32)",
    )
    p.add_argument("--fidelities", nargs="+", default=DEFAULT_FIDELITIES, choices=list(FIDELITY_MAP.keys()))
    p.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="Measurement runs after warmup")
    p.add_argument("--warmup", type=int, default=DEFAULT_WARMUP_RUNS)
    p.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory for CSV and plots")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output)

    device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(2, 2),
        l1_small_size=24576,
    )

    results = []
    configs = [(bs, fi) for bs in args.batch_sizes for fi in args.fidelities]

    for i, (batch_size, fidelity) in enumerate(configs):
        total_batch = batch_size * device.get_num_devices()
        print(f"\n[{i+1}/{len(configs)}] batch={total_batch}  fidelity={fidelity}")
        try:
            r = run_one(
                device,
                batch_size=batch_size,
                fidelity_name=fidelity,
                num_runs=args.runs,
                num_warmup=args.warmup,
            )
            print(f"  init        {r.init_time_s:.2f}s")
            print(f"  compile     {r.compile_time_s:.2f}s")
            print(
                f"  latency     {r.mean_latency_ms:.1f} ± {r.std_latency_ms:.1f} ms  "
                f"[min {r.min_latency_ms:.1f}  max {r.max_latency_ms:.1f}]"
            )
            print(f"  throughput  {r.throughput_img_s:.0f} img/s")
            results.append(r)
        except Exception as e:
            print(f"  SKIPPED: {e}")

    ttnn.close_mesh_device(device)

    if not results:
        print("No results collected.")
        return

    save_csv(results, out_dir / "resnet50_results.csv")
    save_plots(results, out_dir)


if __name__ == "__main__":
    main()
