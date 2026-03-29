#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
SentenceBERT (bert-base backbone) performance benchmark on a single Blackhole TT device.

Model: emrecan/bert-base-turkish-cased-mean-nli-stsb-tr  (BERT-base)
Uses the Blackhole-optimised 2-CQ trace runner.

Usage:
    cd /home/ttuser/noble_workspace/tt-metal
    source python_env/bin/activate
    export ARCH_NAME=blackhole
    export TT_METAL_HOME=$(pwd)
    export PYTHONPATH=$(pwd)
    export TT_METAL_DISABLE_FABRIC_TWO_ERISC=1

    python benchmark_bert.py
    python benchmark_bert.py --runs 20 --output results/bert
    python benchmark_bert.py --batch-sizes 8 --seq-lengths 128 384
"""

import argparse
import csv
import statistics
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import matplotlib.pyplot as plt

import ttnn
from models.demos.sentence_bert.runner.performant_runner import SentenceBERTPerformantRunner


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_BATCH_SIZES = [8]
DEFAULT_SEQ_LENGTHS = [384]
DEFAULT_RUNS = 10  # inference runs after warmup
DEFAULT_WARMUP_RUNS = 2
DEFAULT_OUTPUT_DIR = "generated/benchmarks/bert"

# Required by the Blackhole 2-CQ trace runner
DEVICE_PARAMS = dict(
    l1_small_size=79104,
    trace_region_size=23887872,
    num_command_queues=2,
)


# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    model: str
    batch_size: int  # total across all devices
    sequence_length: int
    act_dtype: str
    weight_dtype: str
    num_devices: int
    # timing (seconds unless noted)
    init_time_s: float = 0.0  # model load + weight transfer
    trace_capture_s: float = 0.0  # 2-CQ trace compilation/capture
    warmup_time_s: float = 0.0
    mean_latency_ms: float = 0.0
    std_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    throughput_seq_s: float = 0.0  # sequences (sentences) per second


# ---------------------------------------------------------------------------
# Benchmark one configuration
# ---------------------------------------------------------------------------


def run_one(
    device,
    batch_size: int,
    seq_length: int,
    act_dtype=ttnn.bfloat16,
    weight_dtype=ttnn.bfloat8_b,
    num_runs: int = DEFAULT_RUNS,
    num_warmup: int = DEFAULT_WARMUP_RUNS,
) -> BenchmarkResult:
    num_devices = device.get_num_devices()
    total_batch = batch_size * num_devices

    result = BenchmarkResult(
        model="sentence-bert-base",
        batch_size=total_batch,
        sequence_length=seq_length,
        act_dtype=str(act_dtype),
        weight_dtype=str(weight_dtype),
        num_devices=num_devices,
    )

    # ---- init (model load + weight transfer to device) ------------------
    t0 = time.perf_counter()
    runner = SentenceBERTPerformantRunner(
        device=device,
        device_batch_size=batch_size,
        sequence_length=seq_length,
        act_dtype=act_dtype,
        weight_dtype=weight_dtype,
        model_location_generator=None,
    )
    ttnn.synchronize_device(device)
    result.init_time_s = time.perf_counter() - t0

    # ---- trace capture (equivalent to op compilation) -------------------
    t0 = time.perf_counter()
    runner._capture_sentencebert_trace_2cqs()
    ttnn.synchronize_device(device)
    result.trace_capture_s = time.perf_counter() - t0

    # ---- warmup (discarded) ---------------------------------------------
    t0 = time.perf_counter()
    for _ in range(num_warmup):
        runner.run()
    ttnn.synchronize_device(device)
    result.warmup_time_s = time.perf_counter() - t0

    # ---- measurement ----------------------------------------------------
    latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        runner.run()
        ttnn.synchronize_device(device)
        latencies.append((time.perf_counter() - t0) * 1e3)  # ms

    result.mean_latency_ms = statistics.mean(latencies)
    result.std_latency_ms = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
    result.min_latency_ms = min(latencies)
    result.max_latency_ms = max(latencies)
    result.throughput_seq_s = total_batch / (result.mean_latency_ms / 1e3)

    runner.release()
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

    labels = [f"bs{r.batch_size}\nseq{r.sequence_length}" for r in results]
    latencies = [r.mean_latency_ms for r in results]
    stds = [r.std_latency_ms for r in results]
    throughputs = [r.throughput_seq_s for r in results]

    x = range(len(results))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("SentenceBERT (BERT-base) – Blackhole Benchmark", fontsize=13)

    # Latency
    bars = ax1.bar(x, latencies, yerr=stds, capsize=4, color="steelblue", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("End-to-end latency (ms)")
    ax1.set_title("Per-batch Latency")
    for bar, val in zip(bars, latencies):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Throughput
    bars2 = ax2.bar(x, throughputs, color="darkorange", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("Throughput (sequences / s)")
    ax2.set_title("Inference Throughput")
    for bar, val in zip(bars2, throughputs):
        ax2.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 5, f"{val:.0f}", ha="center", va="bottom", fontsize=8
        )

    plt.tight_layout()
    plot_path = out_dir / "bert_benchmark.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plot saved → {plot_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="SentenceBERT benchmark on Blackhole")
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=DEFAULT_BATCH_SIZES, help="Per-device batch sizes")
    p.add_argument("--seq-lengths", type=int, nargs="+", default=DEFAULT_SEQ_LENGTHS, help="Sequence lengths to sweep")
    p.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="Measurement runs after warmup")
    p.add_argument("--warmup", type=int, default=DEFAULT_WARMUP_RUNS)
    p.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory for CSV and plots")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output)

    device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(2, 2),
        **DEVICE_PARAMS,
    )

    results = []
    configs = [(bs, sl) for bs in args.batch_sizes for sl in args.seq_lengths]

    for i, (batch_size, seq_length) in enumerate(configs):
        total_batch = batch_size * device.get_num_devices()
        print(f"\n[{i+1}/{len(configs)}] batch={total_batch}  seq_len={seq_length}")
        try:
            r = run_one(
                device,
                batch_size=batch_size,
                seq_length=seq_length,
                num_runs=args.runs,
                num_warmup=args.warmup,
            )
            print(f"  init          {r.init_time_s:.2f}s")
            print(f"  trace capture {r.trace_capture_s:.2f}s")
            print(
                f"  latency       {r.mean_latency_ms:.1f} ± {r.std_latency_ms:.1f} ms  "
                f"[min {r.min_latency_ms:.1f}  max {r.max_latency_ms:.1f}]"
            )
            print(f"  throughput    {r.throughput_seq_s:.0f} seq/s")
            results.append(r)
        except Exception as e:
            print(f"  SKIPPED: {e}")

    ttnn.close_mesh_device(device)

    if not results:
        print("No results collected.")
        return

    save_csv(results, out_dir / "bert_results.csv")
    save_plots(results, out_dir)


if __name__ == "__main__":
    main()
