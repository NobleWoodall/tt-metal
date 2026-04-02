#!/usr/bin/env python3
"""
plot_training.py — interactive HTML plot of TT-Metal nano_gpt training metrics.

Usage:
    python plot_training.py                          # uses training_log.csv
    python plot_training.py --csv my_run.csv
    python plot_training.py --csv run_a.csv run_b.csv   # overlay multiple runs
    python plot_training.py --time-budget 600        # adds vertical line at 600s
    python plot_training.py --target-loss 1.5        # adds horizontal line at target loss
    python plot_training.py --output comparison.html

CSV format expected (produced by csv_logger.hpp):
    run_label, step, wall_time_s, step_time_ms, optimizer_time_ms, train_loss
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLORS = [
    "#2563eb",  # blue
    "#dc2626",  # red
    "#16a34a",  # green
    "#d97706",  # amber
    "#7c3aed",  # violet
    "#0891b2",  # cyan
]


def load_csvs(paths: list[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        df.columns = df.columns.str.strip()
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def smooth(series: pd.Series, window: int = 10) -> pd.Series:
    return series.rolling(window=window, min_periods=1, center=True).mean()


def main():
    parser = argparse.ArgumentParser(description="Plot TT-Metal nano_gpt training metrics")
    parser.add_argument("--csv", nargs="+", default=["training_log.csv"])
    parser.add_argument("--output", default="training_plots.html")
    parser.add_argument("--title", default="TT-Metal — Optimizer Comparison")
    parser.add_argument("--smooth-window", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--time-budget", type=float, default=None)
    parser.add_argument("--target-loss", type=float, default=None)
    args = parser.parse_args()

    missing = [p for p in args.csv if not Path(p).exists()]
    if missing:
        print(f"ERROR: CSV file(s) not found: {missing}", file=sys.stderr)
        sys.exit(1)

    df = load_csvs(args.csv)
    runs = df["run_label"].unique()
    print(f"Loaded {len(df)} rows across {len(runs)} run(s): {list(runs)}")

    fig = make_subplots(
        rows=1,
        cols=5,
        subplot_titles=[
            "Loss vs Step",
            "Loss vs Wall-Clock Time",
            "Full Step Time",
            "Optimizer Step Time",
            "Tokens / sec",
        ],
        horizontal_spacing=0.06,
    )

    for i, run in enumerate(runs):
        color = COLORS[i % len(COLORS)]
        rdf = df[df["run_label"] == run].sort_values("step").reset_index(drop=True)
        loss_smooth = smooth(rdf["train_loss"], args.smooth_window)
        step_smooth = smooth(rdf["step_time_ms"], args.smooth_window)
        opt_smooth = smooth(rdf["optimizer_time_ms"], args.smooth_window)
        tok_smooth = smooth(rdf["tokens_per_sec"], args.smooth_window) if "tokens_per_sec" in rdf.columns else None

        show_legend = True  # only first trace per run shows in legend

        # ── Plot 1: Loss vs Step ────────────────────────────────────────────
        fig.add_trace(
            go.Scatter(
                x=rdf["step"],
                y=rdf["train_loss"],
                mode="lines",
                name=run,
                legendgroup=run,
                line=dict(color=color, width=0.5),
                opacity=0.3,
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=rdf["step"],
                y=loss_smooth,
                mode="lines",
                name=run,
                legendgroup=run,
                line=dict(color=color, width=2),
                showlegend=show_legend,
                hovertemplate=f"<b>{run}</b><br>step=%{{x}}<br>loss=%{{y:.4f}}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # ── Plot 2: Loss vs Wall-Clock Time ────────────────────────────────
        fig.add_trace(
            go.Scatter(
                x=rdf["wall_time_s"],
                y=rdf["train_loss"],
                mode="lines",
                name=run,
                legendgroup=run,
                line=dict(color=color, width=0.5),
                opacity=0.3,
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=rdf["wall_time_s"],
                y=loss_smooth,
                mode="lines",
                name=run,
                legendgroup=run,
                line=dict(color=color, width=2),
                showlegend=False,
                hovertemplate=f"<b>{run}</b><br>t=%{{x:.1f}}s<br>loss=%{{y:.4f}}<extra></extra>",
            ),
            row=1,
            col=2,
        )

        # ── Plot 3: Full Step Time ──────────────────────────────────────────
        fig.add_trace(
            go.Scatter(
                x=rdf["step"],
                y=rdf["step_time_ms"],
                mode="lines",
                name=run,
                legendgroup=run,
                line=dict(color=color, width=0.5),
                opacity=0.3,
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=3,
        )
        fig.add_trace(
            go.Scatter(
                x=rdf["step"],
                y=step_smooth,
                mode="lines",
                name=run,
                legendgroup=run,
                line=dict(color=color, width=2),
                showlegend=False,
                hovertemplate=f"<b>{run}</b><br>step=%{{x}}<br>step_time=%{{y:.1f}}ms<extra></extra>",
            ),
            row=1,
            col=3,
        )

        # ── Plot 4: Optimizer Step Time ─────────────────────────────────────
        fig.add_trace(
            go.Scatter(
                x=rdf["step"],
                y=rdf["optimizer_time_ms"],
                mode="lines",
                name=run,
                legendgroup=run,
                line=dict(color=color, width=0.5),
                opacity=0.3,
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=4,
        )
        fig.add_trace(
            go.Scatter(
                x=rdf["step"],
                y=opt_smooth,
                mode="lines",
                name=run,
                legendgroup=run,
                line=dict(color=color, width=2),
                showlegend=False,
                hovertemplate=f"<b>{run}</b><br>step=%{{x}}<br>opt_time=%{{y:.1f}}ms<extra></extra>",
            ),
            row=1,
            col=4,
        )

        # ── Plot 5: Tokens / sec ────────────────────────────────────────────
        if tok_smooth is not None:
            fig.add_trace(
                go.Scatter(
                    x=rdf["step"],
                    y=rdf["tokens_per_sec"],
                    mode="lines",
                    name=run,
                    legendgroup=run,
                    line=dict(color=color, width=0.5),
                    opacity=0.3,
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=5,
            )
            fig.add_trace(
                go.Scatter(
                    x=rdf["step"],
                    y=tok_smooth,
                    mode="lines",
                    name=run,
                    legendgroup=run,
                    line=dict(color=color, width=2),
                    showlegend=False,
                    hovertemplate=f"<b>{run}</b><br>step=%{{x}}<br>tok/s=%{{y:.0f}}<extra></extra>",
                ),
                row=1,
                col=5,
            )

    # ── Reference lines ─────────────────────────────────────────────────────
    if args.target_loss is not None:
        for col in [1, 2]:
            fig.add_hline(
                y=args.target_loss,
                line=dict(color="gray", dash="dash", width=1),
                annotation_text=f"target={args.target_loss}",
                annotation_position="bottom right",
                row=1,
                col=col,
            )

    if args.time_budget is not None:
        fig.add_vline(
            x=args.time_budget,
            line=dict(color="black", dash="dashdot", width=1.2),
            annotation_text=f"budget={args.time_budget}s",
            row=1,
            col=2,
        )

    # Compilation warmup band on the timing and throughput plots
    for col in [3, 4, 5]:
        fig.add_vrect(
            x0=0,
            x1=args.warmup_steps,
            fillcolor="orange",
            opacity=0.1,
            line_width=0,
            annotation_text="warmup",
            annotation_position="top left",
            row=1,
            col=col,
        )

    # ── Axis labels ─────────────────────────────────────────────────────────
    fig.update_xaxes(title_text="Optimizer Step", row=1, col=1)
    fig.update_xaxes(title_text="Wall-Clock Time (s)", row=1, col=2)
    fig.update_xaxes(title_text="Optimizer Step", row=1, col=3)
    fig.update_xaxes(title_text="Optimizer Step", row=1, col=4)
    fig.update_xaxes(title_text="Optimizer Step", row=1, col=5)

    fig.update_yaxes(title_text="Train Loss", row=1, col=1)
    fig.update_yaxes(title_text="Train Loss", row=1, col=2)
    fig.update_yaxes(title_text="Time (ms)", row=1, col=3)
    fig.update_yaxes(title_text="Time (ms)", row=1, col=4)
    fig.update_yaxes(title_text="Tokens / sec", row=1, col=5)

    fig.update_layout(
        title=dict(text=args.title, font=dict(size=15)),
        height=500,
        width=1750,
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="left", x=0),
        hovermode="x unified",
        template="plotly_white",
        font=dict(family="monospace"),
    )

    output = args.output if args.output.endswith(".html") else args.output.replace(".png", ".html")
    fig.write_html(output, include_plotlyjs="cdn")
    print(f"Saved → {output}  (open in any browser)")

    # ── Summary stats ────────────────────────────────────────────────────────
    print("\n── Summary ──────────────────────────────────────────────────────")
    for run in runs:
        rdf = df[df["run_label"] == run].sort_values("step")
        total_steps = rdf["step"].max()
        total_time = rdf["wall_time_s"].max()
        final_loss = rdf["train_loss"].iloc[-1]
        w = args.warmup_steps
        steady = rdf[rdf["step"] > w]
        median_step_ms = rdf["step_time_ms"].median()
        steady_step_ms = steady["step_time_ms"].median()
        median_opt_ms = rdf["optimizer_time_ms"].median()
        steady_opt_ms = steady["optimizer_time_ms"].median()
        print(f"  [{run}]")
        print(f"    Steps:              {total_steps}")
        print(f"    Total wall time:    {total_time:.1f}s  ({total_time/60:.1f}min)")
        print(f"    Final train loss:   {final_loss:.4f}")
        print(f"    Median step time:   {median_step_ms:.1f}ms  (all steps)")
        print(f"    Steady step time:   {steady_step_ms:.1f}ms  (steps > {w}, post-compile)")
        print(f"    Median opt time:    {median_opt_ms:.1f}ms  (all steps)")
        print(f"    Steady opt time:    {steady_opt_ms:.1f}ms  (steps > {w}, post-compile)")
        if "tokens_per_sec" in rdf.columns:
            steady_tok = steady["tokens_per_sec"].median()
            print(f"    Steady tok/sec:     {steady_tok:.0f}  (steps > {w}, post-compile)")
        if args.target_loss is not None:
            loss_smooth = smooth(rdf["train_loss"])
            crossed = rdf[loss_smooth <= args.target_loss]
            if not crossed.empty:
                print(
                    f"    Reached loss={args.target_loss} at step {crossed['step'].iloc[0]}, t={crossed['wall_time_s'].iloc[0]:.1f}s"
                )
            else:
                print(f"    Did not reach target loss={args.target_loss}")


if __name__ == "__main__":
    main()
