#!/usr/bin/env python3
# Copyright 2026 – Contextual Progressive Token Dropping project.
# Apache License 2.0.
"""
Prepares a BERT-style MLM CSV dataset from a HuggingFace text dataset.

Downloads wikitext-103-v1 (or any HuggingFace dataset), tokenizes with
bert-base-uncased, chunks into fixed-length sequences, applies 15% random
MLM masking (80% [MASK] / 10% random / 10% unchanged), and saves as CSV.

CSV columns:
  input_ids      – JSON list[int], masked input token ids
  attention_mask – JSON list[int], 1 for real tokens 0 for padding
  labels         – JSON list[int], -100 for non-masked, original id for masked

Usage:
  python scripts/prepare_hf_data.py \
      --output_csv /data/umityilmaz/token_drop_l2/data/wikitext_mlm.csv \
      --seq_len 64 \
      --max_samples 400000 \
      --dataset wikitext \
      --dataset_config wikitext-103-v1 \
      --split train
"""

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

# ── progress helper ─────────────────────────────────────────────────────────
def _bar(done, total, width=40):
    frac = done / max(total, 1)
    filled = int(width * frac)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {done:>7,}/{total:,}  ({100*frac:.1f}%)"


# ── MLM masking ──────────────────────────────────────────────────────────────
MASK_TOKEN_ID   = 103   # [MASK]
CLS_TOKEN_ID    = 101   # [CLS]
SEP_TOKEN_ID    = 102   # [SEP]
PAD_TOKEN_ID    = 0     # [PAD]
UNK_TOKEN_ID    = 100   # [UNK]

# Never mask special tokens
_SPECIAL = {CLS_TOKEN_ID, SEP_TOKEN_ID, PAD_TOKEN_ID, UNK_TOKEN_ID}

MASK_PROB       = 0.15   # 15% of eligible tokens chosen for prediction
MASK_REPLACE    = 0.80   # 80% of chosen → [MASK]
RANDOM_REPLACE  = 0.10   # 10% of chosen → random vocab token


def apply_mlm_masking(input_ids: list, vocab_size: int, rng: random.Random):
    """
    Returns (masked_input_ids, labels).

    labels[i] = original_id  if token i was selected for prediction
    labels[i] = -100          otherwise
    """
    n = len(input_ids)
    labels = [-100] * n
    masked = list(input_ids)

    for i, tok in enumerate(input_ids):
        if tok in _SPECIAL:
            continue
        if rng.random() >= MASK_PROB:
            continue

        # This position is selected for prediction
        labels[i] = tok
        r = rng.random()
        if r < MASK_REPLACE:
            masked[i] = MASK_TOKEN_ID
        elif r < MASK_REPLACE + RANDOM_REPLACE:
            # random token (avoid special tokens)
            masked[i] = rng.randint(999, vocab_size - 1)
        # else: keep original (10%)

    return masked, labels


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Prepare HuggingFace MLM dataset → CSV")
    ap.add_argument("--output_csv",      required=True,
                    help="Path to output CSV file")
    ap.add_argument("--seq_len",         type=int, default=64,
                    help="Fixed sequence length including [CLS] and [SEP] (default: 64)")
    ap.add_argument("--max_samples",     type=int, default=400_000,
                    help="Maximum number of sequences to write (default: 400000)")
    ap.add_argument("--dataset",         default="wikitext",
                    help="HuggingFace dataset name (default: wikitext)")
    ap.add_argument("--dataset_config",  default="wikitext-103-v1",
                    help="Dataset config/subset name (default: wikitext-103-v1)")
    ap.add_argument("--split",           default="train",
                    help="Dataset split (default: train)")
    ap.add_argument("--tokenizer",       default="bert-base-uncased",
                    help="HuggingFace tokenizer (default: bert-base-uncased)")
    ap.add_argument("--seed",            type=int, default=42,
                    help="Random seed (default: 42)")
    ap.add_argument("--stride",          type=int, default=None,
                    help="Sliding window stride; defaults to seq_len (no overlap)")
    ap.add_argument("--cache_dir",       default=None,
                    help="Optional HuggingFace cache directory")
    args = ap.parse_args()

    stride = args.stride if args.stride is not None else args.seq_len
    max_content = args.seq_len - 2   # room for [CLS] and [SEP]

    rng = random.Random(args.seed)

    # ── Load tokenizer ────────────────────────────────────────────────────────
    print(f"Loading tokenizer: {args.tokenizer}", flush=True)
    from transformers import BertTokenizerFast
    tokenizer = BertTokenizerFast.from_pretrained(
        args.tokenizer, cache_dir=args.cache_dir)
    vocab_size = tokenizer.vocab_size
    print(f"  Vocab size: {vocab_size:,}", flush=True)

    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f"\nLoading dataset: {args.dataset} / {args.dataset_config} / {args.split}",
          flush=True)
    from datasets import load_dataset
    ds = load_dataset(
        args.dataset, args.dataset_config,
        split=args.split,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    )
    print(f"  Raw examples: {len(ds):,}", flush=True)

    # ── Output file ───────────────────────────────────────────────────────────
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput: {out_path}", flush=True)
    print(f"  seq_len={args.seq_len}  stride={stride}  "
          f"max_samples={args.max_samples:,}  mask_prob={MASK_PROB}", flush=True)

    written   = 0
    skipped   = 0
    processed = 0
    token_buf = []   # rolling buffer of token ids (no special tokens)

    print("\nProcessing…", flush=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["input_ids", "attention_mask", "labels"])
        writer.writeheader()

        for example in ds:
            if written >= args.max_samples:
                break

            text = example.get("text", "")
            if not text or not text.strip():
                skipped += 1
                continue

            # Tokenize (no special tokens; we add them per chunk)
            ids = tokenizer.encode(text, add_special_tokens=False)
            token_buf.extend(ids)

            # Emit all complete chunks from the buffer
            while len(token_buf) >= max_content and written < args.max_samples:
                chunk = token_buf[:max_content]

                # Build input_ids with [CLS] ... [SEP]
                raw_ids = [CLS_TOKEN_ID] + chunk + [SEP_TOKEN_ID]
                attn    = [1] * len(raw_ids)

                # Pad if shorter (shouldn't happen here, but safety)
                pad_len = args.seq_len - len(raw_ids)
                raw_ids += [PAD_TOKEN_ID] * pad_len
                attn    += [0] * pad_len

                # Apply MLM masking
                masked_ids, labels = apply_mlm_masking(raw_ids, vocab_size, rng)

                # Only write sequences that actually have at least one mask
                if all(l == -100 for l in labels):
                    # Re-try masking once (rare edge case)
                    masked_ids, labels = apply_mlm_masking(raw_ids, vocab_size, rng)

                writer.writerow({
                    "input_ids":      json.dumps(masked_ids),
                    "attention_mask": json.dumps(attn),
                    "labels":         json.dumps(labels),
                })
                written += 1

                # Advance buffer by stride
                token_buf = token_buf[stride:]

                # Progress bar every 10k rows
                if written % 10_000 == 0:
                    print(f"  {_bar(written, args.max_samples)}", flush=True)

            processed += 1

    # ── Stats ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print(f"  Done!", flush=True)
    print(f"  Written  : {written:,} sequences", flush=True)
    print(f"  Skipped  : {skipped:,} empty examples", flush=True)
    print(f"  Seq len  : {args.seq_len}", flush=True)
    print(f"  Mask prob: {MASK_PROB:.0%}", flush=True)
    size_mb = out_path.stat().st_size / 1_048_576
    print(f"  File size: {size_mb:.1f} MB", flush=True)
    print(f"  Output   : {out_path}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
