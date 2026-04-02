#!/usr/bin/env python3
"""
prepare_tinyllama_data.py — download WikiText-103 and tokenize for TinyLlama.

Produces data/tinyllama_wikitext103.yaml (BPE tokens) that the nano_gpt binary
reads when configs specify:
    tokenizer_type: "bpe"
    data_path: "data/tinyllama_wikitext103.yaml"

Run once from the tt-train/ directory:
    python prepare_tinyllama_data.py

Requires:
    pip install datasets tokenizers pyyaml
"""

import sys
import yaml
from pathlib import Path

TOKEN_CAP = 5_000_000  # ~2.5× a single 3000-step run (1 batch × 2048 seq_len)
TOKENIZER_JSON = Path("data/tinyllama-tokenizer.json")
OUTPUT_YAML = Path("data/tinyllama_wikitext103.yaml")


def main():
    if not TOKENIZER_JSON.exists():
        print(f"ERROR: {TOKENIZER_JSON} not found — run from tt-train/", file=sys.stderr)
        sys.exit(1)

    try:
        from datasets import load_dataset
        from tokenizers import Tokenizer
    except ImportError:
        print("ERROR: run  pip install datasets tokenizers pyyaml  first", file=sys.stderr)
        sys.exit(1)

    print("Downloading WikiText-103 (raw, train split) …")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)

    print(f"Tokenizing with {TOKENIZER_JSON} (capped at {TOKEN_CAP:,} tokens) …")
    tokenizer = Tokenizer.from_file(str(TOKENIZER_JSON))
    vocab_size = tokenizer.get_vocab_size()

    tokens: list[int] = []
    for row in ds:
        line = row["text"].strip()
        if not line:
            continue
        tokens.extend(tokenizer.encode(line).ids)
        if len(tokens) >= TOKEN_CAP:
            tokens = tokens[:TOKEN_CAP]
            break

    print(f"Got {len(tokens):,} tokens  (vocab_size={vocab_size})")

    print(f"Writing {OUTPUT_YAML} …")
    OUTPUT_YAML.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_YAML, "w", encoding="utf-8") as f:
        yaml.dump(
            {"tokenizer_vocab_size": vocab_size, "data_length": len(tokens), "tokens": tokens},
            f,
            default_flow_style=True,
        )

    size_mb = OUTPUT_YAML.stat().st_size / 1024**2
    print(f"Done → {OUTPUT_YAML}  ({size_mb:.1f} MB)")
    print("Update TinyLlama configs with:")
    print("    tokenizer_type: bpe")
    print(f"    data_path: {OUTPUT_YAML}")


if __name__ == "__main__":
    main()
