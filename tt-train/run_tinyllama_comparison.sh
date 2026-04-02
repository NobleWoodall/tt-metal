#!/bin/bash
# Run all 5 TinyLlama optimizer variants back-to-back, logging into one CSV.
# Run this from the tt-train/ directory.

export TT_METAL_LOGGER_LEVEL=ERROR

CSV="tinyllama_log.csv"
BIN="../build/tt-train/sources/examples/nano_gpt/nano_gpt"

rm -f "$CSV"

# echo "=== TinyLlama AdamW (bf16 buffers) ==="
# $BIN --config configs/training_configs/training_shakespeare_tinyllama_adamw.yaml \
#      --csv "$CSV" --run-label "adamw"

echo "=== TinyLlama AdamW + stochastic rounding ==="
$BIN --config configs/training_configs/training_shakespeare_tinyllama_adamw_stoch.yaml \
     --csv "$CSV" --run-label "adamw_stoch"

# echo "=== TinyLlama AdamWFullPrecision (fp32 buffers) ==="
# $BIN --config configs/training_configs/training_shakespeare_tinyllama_adamw_fp32.yaml \
#      --csv "$CSV" --run-label "adamw_fp32"

echo "=== TinyLlama MuonComposite ==="
$BIN --config configs/training_configs/training_shakespeare_tinyllama_muon.yaml \
     --csv "$CSV" --run-label "muon"

# echo "=== TinyLlama SGD (fused default) ==="
# $BIN --config configs/training_configs/training_shakespeare_tinyllama_sgd_fused.yaml \
#      --csv "$CSV" --run-label "sgd_fused"

echo "=== Done. Plotting... ==="
python plot_training.py --csv "$CSV" --output tinyllama_plots.html
