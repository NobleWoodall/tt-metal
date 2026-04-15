#!/bin/bash
# Weak-scaling benchmark: global batch 256 for all runs.
# 1 BH: 256 per device.  4 BH DDP: 64 per device x 4 = 256 global.
# At the same per-device compute, 4 BH should deliver ~4x tokens/sec.
# Run this from the tt-train/ directory.

export TT_METAL_LOGGER_LEVEL=ERROR
export TT_METAL_DISABLE_FABRIC_TWO_ERISC=1
export TT_MESH_GRAPH_DESC_PATH="${TT_METAL_HOME}/tt_metal/fabric/mesh_graph_descriptors/bh_lb_1x4_ring_mesh_graph_descriptor.textproto"

CSV="nanogpt_weakscaling_log.csv"
BIN="../build/tt-train/sources/examples/nano_gpt/nano_gpt"

rm -f "$CSV"

# --- AdamW + stochastic rounding ---

echo "=== NanoGPT AdamW + stochastic rounding  1 BH  batch=256 ==="
$BIN --config configs/training_configs/training_shakespeare_nanogpt_adamw_stoch_1bh_256.yaml \
     --csv "$CSV" --run-label "adamw_1bh_256"

echo "=== NanoGPT AdamW + stochastic rounding  4 BH DDP  batch=256 global ==="
$BIN --config configs/training_configs/training_shakespeare_nanogpt_adamw_stoch_4bh_256.yaml \
     --csv "$CSV" --run-label "adamw_4bh_256"

# --- Muon ---

echo "=== NanoGPT Muon  1 BH  batch=256 ==="
$BIN --config configs/training_configs/training_shakespeare_nanogpt_muon_1bh_256.yaml \
     --csv "$CSV" --run-label "muon_1bh_256"

echo "=== NanoGPT Muon  4 BH DDP  batch=256 global ==="
$BIN --config configs/training_configs/training_shakespeare_nanogpt_muon_4bh_256.yaml \
     --csv "$CSV" --run-label "muon_4bh_256"

# --- SGD (fused) ---

echo "=== NanoGPT SGD fused  1 BH  batch=256 ==="
$BIN --config configs/training_configs/training_shakespeare_nanogpt_sgd_1bh_256.yaml \
     --csv "$CSV" --run-label "sgd_1bh_256"

echo "=== NanoGPT SGD fused  4 BH DDP  batch=256 global ==="
$BIN --config configs/training_configs/training_shakespeare_nanogpt_sgd_4bh_256.yaml \
     --csv "$CSV" --run-label "sgd_4bh_256"

echo "=== Done. Plotting... ==="

python plot_training.py --csv "$CSV" \
    --runs adamw_1bh_256 adamw_4bh_256 \
    --title "TT-Metal — AdamW: 1 BH vs 4 BH DDP (batch=256 global)" \
    --output nanogpt_weakscaling_adamw.html

python plot_training.py --csv "$CSV" \
    --runs muon_1bh_256 muon_4bh_256 \
    --title "TT-Metal — Muon: 1 BH vs 4 BH DDP (batch=256 global)" \
    --output nanogpt_weakscaling_muon.html

python plot_training.py --csv "$CSV" \
    --runs sgd_1bh_256 sgd_4bh_256 \
    --title "TT-Metal — SGD: 1 BH vs 4 BH DDP (batch=256 global)" \
    --output nanogpt_weakscaling_sgd.html
