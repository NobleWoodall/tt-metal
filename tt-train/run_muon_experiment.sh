#!/bin/bash
# Muon scaling experiment: 1 BH baseline vs 4 BH DDP across batch sizes (TinyLlama).
#
# Run from tt-train/:
#   bash run_muon_experiment.sh
#
# Set STEPS=2 for a quick sanity check, STEPS=2000 for a full run.

set -euo pipefail
export TT_METAL_LOGGER_LEVEL=ERROR
export TT_METAL_DISABLE_FABRIC_TWO_ERISC=1
export TT_MESH_GRAPH_DESC_PATH="${TT_METAL_HOME}/tt_metal/fabric/mesh_graph_descriptors/bh_lb_1x4_ring_mesh_graph_descriptor.textproto"

MODEL="tinyllama"
STEPS=2000           # set to 2000 for full run
CSV="results/${MODEL}_muon_scaling.csv"
mkdir -p results

SCRIPT="python run_experiment.py"

# Batch sizes for 4 BH DDP runs (global batch, must be divisible by 4)
BATCH_SIZES_4BH=(8 16 32 64 128)

rm -f "$CSV"

# --- 1 BH baseline (single device, batch=8) ---
echo ""
echo "=== Muon  1 BH  batch=8 ==="
$SCRIPT --config "configs/training_configs/experiment_tinyllama_muon_1bh.yaml" \
        --csv "$CSV" \
        --run-label "muon_1bh_bs8" \
        --steps "$STEPS"

# --- 4 BH DDP at multiple batch sizes ---
for BS in "${BATCH_SIZES_4BH[@]}"; do
    echo ""
    echo "=== Muon  4 BH DDP  batch=$BS ==="
    $SCRIPT --config "configs/training_configs/experiment_tinyllama_muon_4bh.yaml" \
            --csv "$CSV" \
            --run-label "muon_4bh_bs${BS}" \
            --batch-size "$BS" \
            --steps "$STEPS"
done

echo ""
echo "=== Done. Plotting... ==="
python plot_training.py \
    --csv "$CSV" \
    --output "results/${MODEL}_muon_scaling.html" \
    --title "${MODEL}: Muon — 1 BH vs 4 BH DDP across batch sizes"
