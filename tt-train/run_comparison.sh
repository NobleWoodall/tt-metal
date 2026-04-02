#!/bin/bash
# Run the full two-level optimizer benchmark suite.
# Runs NanoGPT (6 optimizers) then TinyLlama (5 optimizers).
# Run this from the tt-train/ directory.

bash run_nanogpt_comparison.sh
bash run_tinyllama_comparison.sh
