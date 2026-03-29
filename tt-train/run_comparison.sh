#!/bin/bash
# Run AdamW, SGD, and SGD Fused back-to-back, logging all into one CSV.
# Run this from the tt-train/ directory.

CSV="training_log.csv"
BIN="./build/sources/examples/nano_gpt/nano_gpt"

rm -f "$CSV"

echo "=== AdamW ==="
$BIN --config configs/training_configs/training_shakespeare_nanogpt_adamw.yaml \
     --csv "$CSV" --run-label "adamw"

echo "=== SGD ==="
$BIN --config configs/training_configs/training_shakespeare_nanogpt_sgd.yaml \
     --csv "$CSV" --run-label "sgd"

echo "=== SGD Fused ==="
$BIN --config configs/training_configs/training_shakespeare_nanogpt_sgd_fused.yaml \
     --csv "$CSV" --run-label "sgd_fused"

echo "=== Done. Plotting... ==="
python plot_training.py --csv "$CSV"
