#!/bin/bash
# Run NanoGPT optimizer variants (1 BH and 4 BH DDP) back-to-back, logging into one CSV.
# Global batch size is 64 for all runs (1 BH: 64 per device, 4 BH: 16 per device x 4).
# 4 BH runs should be ~4x faster per step at the same global batch size.
# Run this from the tt-train/ directory.

export TT_METAL_LOGGER_LEVEL=ERROR
export TT_METAL_DISABLE_FABRIC_TWO_ERISC=1
export TT_MESH_GRAPH_DESC_PATH="${TT_METAL_HOME}/tt_metal/fabric/mesh_graph_descriptors/bh_lb_1x4_ring_mesh_graph_descriptor.textproto"

CSV="nanogpt_log.csv"
BIN="../build/tt-train/sources/examples/nano_gpt/nano_gpt"

rm -f "$CSV"

# --- AdamW + stochastic rounding ---

echo "=== NanoGPT AdamW + stochastic rounding  1 BH ==="
$BIN --config configs/training_configs/training_shakespeare_nanogpt_adamw_stoch.yaml \
     --csv "$CSV" --run-label "adamw_stoch_1bh"

echo "=== NanoGPT AdamW + stochastic rounding  4 BH DDP ==="
$BIN --config configs/training_configs/training_shakespeare_nanogpt_adamw_stoch_4bh.yaml \
     --csv "$CSV" --run-label "adamw_stoch_4bh"

# --- Muon ---

echo "=== NanoGPT Muon  1 BH ==="
$BIN --config configs/training_configs/training_shakespeare_nanogpt_muon.yaml \
     --csv "$CSV" --run-label "muon_1bh"

echo "=== NanoGPT Muon  4 BH DDP ==="
$BIN --config configs/training_configs/training_shakespeare_nanogpt_muon_4bh.yaml \
     --csv "$CSV" --run-label "muon_4bh"

# --- SGD (fused) ---

echo "=== NanoGPT SGD fused  1 BH ==="
$BIN --config configs/training_configs/training_shakespeare_nanogpt_sgd.yaml \
     --csv "$CSV" --run-label "sgd_1bh"

echo "=== NanoGPT SGD fused  4 BH DDP ==="
$BIN --config configs/training_configs/training_shakespeare_nanogpt_sgd_4bh.yaml \
     --csv "$CSV" --run-label "sgd_4bh"


echo "=== Done. Plotting... ==="

python plot_training.py --csv "$CSV" \
    --runs sgd_1bh sgd_4bh \
    --title "TT-Metal — SGD (fused): 1 BH vs 4 BH DDP" \
    --output nanogpt_plots_sgd.html

python plot_training.py --csv "$CSV" \
    --runs adamw_stoch_1bh adamw_stoch_4bh \
    --title "TT-Metal — AdamW + Stochastic Rounding: 1 BH vs 4 BH DDP" \
    --output nanogpt_plots_adamw.html

python plot_training.py --csv "$CSV" \
    --runs muon_1bh muon_4bh \
    --title "TT-Metal — Muon: 1 BH vs 4 BH DDP" \
    --output nanogpt_plots_muon.html
