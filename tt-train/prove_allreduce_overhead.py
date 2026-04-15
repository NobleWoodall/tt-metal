#!/usr/bin/env python3
"""
prove_allreduce_overhead.py

Sweeps batch sizes on 1 BH and 4 BH DDP and plots step time vs per-chip batch size.

Expected result:
  1BH  → roughly linear (step time ∝ batch size)
  4BH  → flattens at small batch (all-reduce dominates), converges to 1BH/4 at large batch

The gap between the 4BH curve and 1BH/4 is the all-reduce overhead.

Usage (from tt-train/):
    python prove_allreduce_overhead.py
    python prove_allreduce_overhead.py --batches 4 8 16 32 64 128
    python prove_allreduce_overhead.py --steps 30 --output allreduce_proof.html
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import yaml

BIN = "../build/tt-train/sources/examples/nano_gpt/nano_gpt"
BASE_CONFIG = "configs/training_configs/training_shakespeare_nanogpt_adamw_stoch.yaml"

BASE_ENV = {
    **os.environ,
    "TT_METAL_LOGGER_LEVEL": "ERROR",
    "TT_METAL_DISABLE_FABRIC_TWO_ERISC": "1",
    "TT_MESH_GRAPH_DESC_PATH": (
        f"{os.environ.get('TT_METAL_HOME', '')}/tt_metal/fabric/mesh_graph_descriptors"
        "/bh_lb_1x4_ring_mesh_graph_descriptor.textproto"
    ),
}


def make_config(batch_size: int, ddp: bool, max_steps: int) -> str:
    cwd = Path.cwd()

    with open(BASE_CONFIG) as f:
        cfg = yaml.safe_load(f)

    # Resolve relative paths to absolute so the binary finds them regardless
    # of where the temp config file lands.
    tc = cfg["training_config"]
    tc["data_path"] = str((cwd / tc["data_path"]).resolve())
    tc["model_config"] = str((cwd / tc["model_config"]).resolve())

    tc["batch_size"] = batch_size
    tc["max_steps"] = max_steps
    tc["model_save_interval"] = max_steps + 1  # never save
    cfg["device_config"]["enable_ddp"] = ddp
    cfg["device_config"]["mesh_shape"] = [1, 4] if ddp else [1, 1]

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, prefix="allreduce_test_")
    yaml.dump(cfg, tmp)
    tmp.close()
    return tmp.name


def run_one(batch_size: int, ddp: bool, max_steps: int, csv_path: str, label: str):
    cfg = make_config(batch_size, ddp, max_steps)
    try:
        cmd = [BIN, "--config", cfg, "--csv", csv_path, "--run-label", label]
        print(f"  running: {label}  (batch={batch_size}, ddp={ddp})")
        result = subprocess.run(cmd, env=BASE_ENV, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ERROR for {label}:\n{result.stderr[-500:]}", file=sys.stderr)
    finally:
        os.unlink(cfg)


def steady_median(df: pd.DataFrame, label: str, warmup: int = 5) -> dict:
    rdf = df[df["run_label"] == label]
    rdf = rdf[rdf["step"] > warmup]
    return {
        "step_time_ms": rdf["step_time_ms"].median(),
        "optimizer_time_ms": rdf["optimizer_time_ms"].median(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batches",
        nargs="+",
        type=int,
        default=[4, 8, 16, 32, 64, 128, 256],
        help="Global batch sizes to sweep (4BH splits by 4 per chip)",
    )
    parser.add_argument("--steps", type=int, default=40, help="Steps per run")
    parser.add_argument("--warmup", type=int, default=5, help="Steps to discard as warmup")
    parser.add_argument("--output", default="allreduce_proof.html")
    args = parser.parse_args()

    csv_path = "allreduce_proof.csv"
    Path(csv_path).unlink(missing_ok=True)

    results = []

    for global_batch in args.batches:
        per_chip_4bh = global_batch // 4
        if per_chip_4bh < 1:
            print(f"Skipping batch={global_batch}: too small to split across 4 chips")
            continue

        label_1bh = f"1bh_b{global_batch}"
        label_4bh = f"4bh_b{global_batch}"

        run_one(global_batch, ddp=False, max_steps=args.steps, csv_path=csv_path, label=label_1bh)
        run_one(global_batch, ddp=True, max_steps=args.steps, csv_path=csv_path, label=label_4bh)

        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        s1 = steady_median(df, label_1bh, args.warmup)
        s4 = steady_median(df, label_4bh, args.warmup)

        results.append(
            {
                "global_batch": global_batch,
                "per_chip_4bh": per_chip_4bh,
                "step_1bh": s1["step_time_ms"],
                "step_4bh": s4["step_time_ms"],
                "opt_1bh": s1["optimizer_time_ms"],
                "opt_4bh": s4["optimizer_time_ms"],
            }
        )
        print(
            f"  batch={global_batch:4d}  1BH={s1['step_time_ms']:.1f}ms  "
            f"4BH={s4['step_time_ms']:.1f}ms  "
            f"ideal_4BH={s1['step_time_ms']/4:.1f}ms"
        )

    if not results:
        print("No results collected.", file=sys.stderr)
        sys.exit(1)

    r = pd.DataFrame(results)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=r["global_batch"],
            y=r["step_1bh"],
            mode="lines+markers",
            name="1 BH (actual)",
            line=dict(color="#2563eb", width=2),
            marker=dict(size=8),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=r["global_batch"],
            y=r["step_4bh"],
            mode="lines+markers",
            name="4 BH DDP (actual)",
            line=dict(color="#dc2626", width=2),
            marker=dict(size=8),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=r["global_batch"],
            y=r["step_1bh"] / 4,
            mode="lines",
            name="1 BH ÷ 4 (ideal 4BH with zero comm)",
            line=dict(color="#2563eb", width=1.5, dash="dash"),
        )
    )

    fig.update_layout(
        title="Step Time vs Global Batch Size — Proving All-Reduce Floor",
        xaxis_title="Global Batch Size (tokens per step)",
        yaxis_title="Median Step Time (ms, post-warmup)",
        template="plotly_white",
        font=dict(family="monospace"),
        width=900,
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        annotations=[
            dict(
                x=0.5,
                y=-0.18,
                xref="paper",
                yref="paper",
                showarrow=False,
                text=(
                    "Gap between red and dashed blue = all-reduce overhead. "
                    "At small batch it dominates; at large batch it shrinks to a rounding error."
                ),
                font=dict(size=11, color="gray"),
            )
        ],
    )

    fig.write_html(args.output, include_plotlyjs="cdn")
    print(f"\nSaved → {args.output}")

    print("\n── Raw numbers ──────────────────────────────────────────────────")
    print(f"{'batch':>6}  {'1BH (ms)':>10}  {'4BH (ms)':>10}  {'ideal (ms)':>10}  {'overhead (ms)':>14}")
    for _, row in r.iterrows():
        ideal = row["step_1bh"] / 4
        overhead = row["step_4bh"] - ideal
        print(
            f"{int(row['global_batch']):>6}  {row['step_1bh']:>10.1f}  "
            f"{row['step_4bh']:>10.1f}  {ideal:>10.1f}  {overhead:>14.1f}"
        )


if __name__ == "__main__":
    main()
