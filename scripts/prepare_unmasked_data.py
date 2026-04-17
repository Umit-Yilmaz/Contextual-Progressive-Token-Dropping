#!/usr/bin/env python3
# Copyright 2026 – Contextual Progressive Token Dropping project.
# Apache License 2.0.
"""
Prepares an UNMASKED tokenized CSV dataset from a HuggingFace text dataset.

Unlike prepare_hf_data.py, this script does NOT apply MLM masking.
It saves raw token IDs so that masking can be applied dynamically
during training (different masks each epoch).

CSV columns:
  input_ids      – JSON list[int], original token ids (NO masking)
  attention_mask – JSON list[int], 1 for real tokens 0 for padding

Usage:
  python scripts/prepare_unmasked_data.py \
      --output_csv /data/umityilmaz/token_drop_v2/data/wikitext_unmasked_512.csv \
      --seq_len 512 \
      --max_samples 400000 \
      --dataset wikitext \
      --dataset_config wikitext-103-v1 \
      --split train
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path


# ── progress helper ─────────────────────────────────────────────────────────
def _bar(done, total, width=40):
    frac = done / max(total, 1)
    filled = int(width * frac)
    bar = "\u2588" * filled + "\u2591" * (width - filled)
    return f"[{bar}] {done:>7,}/{total:,}  ({100*frac:.1f}%)"


# ── Special token IDs (bert-base-uncased) ───────────────────────────────────
CLS_TOKEN_ID = 101   # [CLS]
SEP_TOKEN_ID = 102   # [SEP]
PAD_TOKEN_ID = 0     # [PAD]


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Prepare HuggingFace UNMASKED tokenized dataset -> CSV")
    ap.add_argument("--output_csv",      required=True,
                    help="Path to output CSV file")
    ap.add_argument("--seq_len",         type=int, default=512,
                    help="Fixed sequence length including [CLS] and [SEP] (default: 512)")
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
    ap.add_argument("--stride",          type=int, default=None,
                    help="Sliding window stride; defaults to seq_len (no overlap)")
    ap.add_argument("--cache_dir",       default=None,
                    help="Optional HuggingFace cache directory")
    args = ap.parse_args()

    stride = args.stride if args.stride is not None else args.seq_len
    max_content = args.seq_len - 2   # room for [CLS] and [SEP]

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
          f"max_samples={args.max_samples:,}  (NO masking applied)", flush=True)

    written   = 0
    skipped   = 0
    token_buf = []   # rolling buffer of token ids (no special tokens)

    print("\nProcessing...", flush=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["input_ids", "attention_mask"])
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

                # Write WITHOUT masking
                writer.writerow({
                    "input_ids":      json.dumps(raw_ids),
                    "attention_mask": json.dumps(attn),
                })
                written += 1

                # Advance buffer by stride
                token_buf = token_buf[stride:]

                # Progress bar every 10k rows
                if written % 10_000 == 0:
                    print(f"  {_bar(written, args.max_samples)}", flush=True)

    # ── Stats ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print(f"  Done!", flush=True)
    print(f"  Written  : {written:,} sequences", flush=True)
    print(f"  Skipped  : {skipped:,} empty examples", flush=True)
    print(f"  Seq len  : {args.seq_len}", flush=True)
    print(f"  Masking  : NONE (apply dynamically during training)", flush=True)
    size_mb = out_path.stat().st_size / 1_048_576
    print(f"  File size: {size_mb:.1f} MB", flush=True)
    print(f"  Output   : {out_path}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
