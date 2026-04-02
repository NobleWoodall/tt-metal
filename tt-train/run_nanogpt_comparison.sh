#!/bin/bash
# Run all 6 NanoGPT optimizer variants back-to-back, logging into one CSV.
# Run this from the tt-train/ directory.

export TT_METAL_LOGGER_LEVEL=ERROR

CSV="nanogpt_log.csv"
BIN="../build/tt-train/sources/examples/nano_gpt/nano_gpt"

rm -f "$CSV"

# echo "=== NanoGPT AdamW (bf16 buffers) ==="
# $BIN --config configs/training_configs/training_shakespeare_nanogpt_adamw.yaml \
#      --csv "$CSV" --run-label "adamw"

echo "=== NanoGPT MuonComposite ==="
$BIN --config configs/training_configs/training_shakespeare_nanogpt_muon.yaml \
     --csv "$CSV" --run-label "muon"

echo "=== NanoGPT AdamW + stochastic rounding ==="
$BIN --config configs/training_configs/training_shakespeare_nanogpt_adamw_stoch.yaml \
     --csv "$CSV" --run-label "adamw_stoch"

# echo "=== NanoGPT AdamWFullPrecision (fp32 buffers) ==="
# $BIN --config configs/training_configs/training_shakespeare_nanogpt_adamw_fp32.yaml \
#      --csv "$CSV" --run-label "adamw_fp32"

# echo "=== NanoGPT SGDComposite ==="
# $BIN --config configs/training_configs/training_shakespeare_nanogpt_sgd.yaml \
#      --csv "$CSV" --run-label "sgd"

# echo "=== NanoGPT SGD (fused default) ==="
# $BIN --config configs/training_configs/training_shakespeare_nanogpt_sgd_fused.yaml \
#      --csv "$CSV" --run-label "sgd_fused"


echo "=== Done. Plotting... ==="
python plot_training.py --csv "$CSV" --output nanogpt_plots.html
