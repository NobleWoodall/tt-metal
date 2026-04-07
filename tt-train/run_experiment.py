#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Muon vs AdamW comparison experiment across batch sizes and DDP.

Run from tt-train/:
    python run_experiment.py \
        --config configs/training_configs/experiment_nanogpt_muon_4bh.yaml \
        --csv results/nanogpt_log.csv \
        --run-label muon_bs64_4bh \
        [--batch-size 64]

Multiple runs append to the same CSV and are plotted together with plot_training.py.
"""

import argparse
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.environ["TT_METAL_HOME"], "tt-train/sources/ttml"))

import numpy as np
import ttml
import ttml.common.muon_optimizer

ttml.common.muon_optimizer.register()

from ttml.common.config import load_config, DeviceConfig, TransformerConfig
from ttml.common.utils import get_tt_metal_runtime_root, initialize_device, create_optimizer, round_up_to_tile, set_seed
from ttml.common.model_factory import TransformerModelFactory
from ttml.common.data import CharTokenizer
from ttml.common.trainer import train, CSVLogger


@dataclass
class TrainCfg:
    """Config bag for trainer.train() — adds seq_len to TrainingConfig fields."""

    batch_size: int
    seq_len: int
    steps: int
    gradient_accumulation_steps: int
    eval_every: int


def main():
    parser = argparse.ArgumentParser(description="Muon vs AdamW experiment")
    parser.add_argument("-c", "--config", required=True, help="Path to training YAML config")
    parser.add_argument("--csv", default="training_log.csv", help="CSV output path")
    parser.add_argument("--run-label", default="run", help="Label for this run in the CSV")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size from config")
    parser.add_argument("--steps", type=int, default=None, help="Override max_steps from config")
    args = parser.parse_args()

    tt_root = get_tt_metal_runtime_root()
    tt_train_root = os.path.join(tt_root, "tt-train")

    yaml_config = load_config(args.config, os.path.join(tt_train_root, "configs/training_configs"))
    tc = yaml_config["training_config"]

    # Override CLI args
    if args.batch_size is not None:
        tc["batch_size"] = args.batch_size
    if args.steps is not None:
        tc["max_steps"] = args.steps

    batch_size = tc.get("batch_size", 64)
    steps = tc.get("max_steps", 2000)
    grad_accum = tc.get("gradient_accumulation_steps", 1)
    eval_every = tc.get("eval_every", 200)
    data_path = tc.get("data_path", os.path.join(tt_train_root, "data/shakespeare.txt"))
    seed = tc.get("seed", 5489)

    device_cfg = DeviceConfig(yaml_config)
    num_devices = device_cfg.total_devices()

    # Load model config to get seq_len
    model_yaml = load_config(tc["model_config"], tt_train_root)
    seq_len = model_yaml.get("transformer_config", {}).get("max_sequence_length", 256)

    # Load and tokenize data
    if data_path.endswith(".yaml") or data_path.endswith(".yml"):
        # Pre-tokenized dataset: {data_length, tokenizer_vocab_size, tokens: [...]}
        import yaml as _yaml

        with open(data_path) as f:
            pretok = _yaml.safe_load(f)
        tokens = np.array(pretok["tokens"], dtype=np.uint32)
        vocab_size = round_up_to_tile(int(pretok["tokenizer_vocab_size"]), 32)
    else:
        with open(data_path) as f:
            text = f.read()
        tokenizer = CharTokenizer(text)
        tokens = np.array(tokenizer.encode(text), dtype=np.uint32)
        vocab_size = round_up_to_tile(tokenizer.vocab_size, 32)

    split = int(len(tokens) * 0.9)
    train_ids = tokens[:split]

    # Patch vocab_size into config for model factory
    yaml_config.setdefault("training_config", {}).setdefault("transformer_config", {})
    yaml_config["training_config"]["transformer_config"]["vocab_size"] = vocab_size

    set_seed(seed)

    # Initialize device (handles fabric automatically for multi-device)
    initialize_device(yaml_config)

    if device_cfg.enable_ddp:
        dist_cfg = ttml.autograd.DistributedConfig()
        dist_cfg.enable_ddp = True
        ttml.autograd.AutoContext.get_instance().initialize_parallelism_context(dist_cfg)

    # Create model and optimizer
    factory = TransformerModelFactory(yaml_config)
    model = factory.create_model()
    optimizer = create_optimizer(model, yaml_config)

    effective_batch = batch_size * grad_accum
    print(f"Model:             {factory.model_type}")
    print(f"Optimizer:         {optimizer.get_name()}")
    print(f"Devices (DDP):     {num_devices}  (ddp={device_cfg.enable_ddp})")
    print(f"Batch per device:  {batch_size // num_devices if device_cfg.enable_ddp else batch_size}")
    print(f"Global batch:      {effective_batch * num_devices if device_cfg.enable_ddp else effective_batch}")
    print(f"Seq len:           {seq_len}")
    print(f"Steps:             {steps}")
    print(f"CSV:               {args.csv}  [{args.run_label}]")
    print()

    if optimizer.get_name() == "MuonWithAdamW":
        optimizer.print_param_groups()
        print()

    cfg = TrainCfg(
        batch_size=batch_size,
        seq_len=seq_len,
        steps=steps,
        gradient_accumulation_steps=grad_accum,
        eval_every=eval_every,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
    csv_logger = CSVLogger(args.csv, args.run_label)

    train(
        cfg,
        model,
        optimizer,
        train_ids,
        use_ddp=device_cfg.enable_ddp,
        csv_logger=csv_logger,
    )

    csv_logger.close()
    ttml.autograd.AutoContext.get_instance().close_device()


if __name__ == "__main__":
    main()
