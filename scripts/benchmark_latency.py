#!/usr/bin/env python3
"""
benchmark_latency.py

Measures forward-pass latency for the three BERT variants:
  - Vanilla BERT
  - TokenDrop BERT
  - Progressive Drop BERT

Uses time.perf_counter() with GPU synchronization (via .numpy() call)
to ensure accurate GPU timing. Performs warmup runs to account for
XLA compilation and lazy weight creation.

Usage:
    python scripts/benchmark_latency.py
    python scripts/benchmark_latency.py --batch_size 64 --seq_len 128 --n_runs 20
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

# ── Path setup (same as train_csv_comparison.py) ─────────────────────────────
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(REPO, 'experiments', 'progressive_contextual_dropping'))
sys.path.insert(0, REPO)

import tensorflow as tf

VOCAB_SIZE = 30522  # bert-base-uncased


class BertMLM(tf.keras.Model):
    """Minimal BertMLM wrapper (same as train_csv_comparison.py)."""

    def __init__(self, encoder, vocab_size, hidden_size, name='bert_mlm'):
        super().__init__(name=name)
        self.encoder   = encoder
        self.mlm_dense = tf.keras.layers.Dense(hidden_size, use_bias=True,
                                               name='mlm_dense')
        self.mlm_norm  = tf.keras.layers.LayerNormalization(
                             epsilon=1e-12, name='mlm_norm')
        self.mlm_proj  = tf.keras.layers.Dense(vocab_size, use_bias=True,
                                               name='mlm_proj')

    def call(self, inputs, training=False):
        enc_inp = {
            'input_word_ids': inputs['input_ids'],
            'input_mask':     inputs['attention_mask'],
            'input_type_ids': tf.zeros_like(inputs['input_ids']),
        }
        enc_out = self.encoder(enc_inp, training=training)
        seq_out = enc_out['sequence_output']
        h       = self.mlm_dense(seq_out)
        h       = tf.nn.gelu(h, approximate=True)
        h       = self.mlm_norm(h)
        logits  = self.mlm_proj(h)
        return logits


def build_models(args):
    """Build all three BERT variants."""
    from official.nlp.modeling.networks import bert_encoder as _bert_enc_mod
    from encoder import TokenDropBertEncoder
    from experiments.progressive_contextual_dropping.encoder import (
        ProgressiveContextualDropEncoder)

    common = dict(
        hidden_size         = args.hidden_size,
        num_layers          = args.num_layers,
        num_attention_heads = args.num_heads,
        output_dropout      = args.dropout_rate,
        attention_dropout   = args.dropout_rate,
    )

    vanilla_enc = _bert_enc_mod.BertEncoder(
        vocab_size=VOCAB_SIZE, inner_dim=args.intermediate_size,
        max_sequence_length=args.max_seq_len, dict_outputs=True, **common)
    vanilla = BertMLM(vanilla_enc, VOCAB_SIZE, args.hidden_size, name='vanilla_mlm')

    tokendrop_enc = TokenDropBertEncoder(
        vocab_size=VOCAB_SIZE, intermediate_size=args.intermediate_size,
        max_sequence_length=args.max_seq_len,
        token_keep_k=args.token_keep_k, **common)
    tokendrop = BertMLM(tokendrop_enc, VOCAB_SIZE, args.hidden_size, name='tokendrop_mlm')

    prog_enc = ProgressiveContextualDropEncoder(
        vocab_size=VOCAB_SIZE, inner_dim=args.intermediate_size,
        max_sequence_length=args.max_seq_len,
        token_keep_k1=args.token_keep_k1, token_keep_k2=args.token_keep_k2,
        token_keep_k3=args.token_keep_k3, **common)
    progressive = BertMLM(prog_enc, VOCAB_SIZE, args.hidden_size, name='progressive_mlm')

    return [('vanilla', vanilla), ('tokendrop', tokendrop), ('progressive', progressive)]


def create_dummy_batch(batch_size, seq_len):
    """Create a dummy input batch."""
    return {
        'input_ids':      tf.constant(np.random.randint(1, VOCAB_SIZE, (batch_size, seq_len)),
                                      dtype=tf.int32),
        'attention_mask':  tf.ones((batch_size, seq_len), dtype=tf.int32),
        'labels':         tf.constant(np.full((batch_size, seq_len), -100), dtype=tf.int32),
    }


def measure_latency(model, batch, n_warmup=3, n_runs=10):
    """Measure forward-pass latency with GPU sync.

    Returns (mean_ms, std_ms, all_times_ms).
    """
    # Warmup: XLA compilation + lazy weight creation
    for _ in range(n_warmup):
        out = model(batch, training=False)
        # Force GPU sync
        tf.reduce_sum(out).numpy()

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        out = model(batch, training=False)
        # Force GPU sync before stopping timer
        tf.reduce_sum(out).numpy()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms

    times = np.array(times)
    return float(times.mean()), float(times.std()), times.tolist()


def main():
    p = argparse.ArgumentParser(description='Benchmark forward-pass latency for 3 BERT variants')
    p.add_argument('--batch_size',        type=int, default=64)
    p.add_argument('--seq_len',           type=int, default=64)
    p.add_argument('--n_warmup',          type=int, default=3)
    p.add_argument('--n_runs',            type=int, default=10)
    p.add_argument('--hidden_size',       type=int, default=256)
    p.add_argument('--num_layers',        type=int, default=4)
    p.add_argument('--num_heads',         type=int, default=4)
    p.add_argument('--intermediate_size', type=int, default=1024)
    p.add_argument('--max_seq_len',       type=int, default=128)
    p.add_argument('--dropout_rate',      type=float, default=0.1)
    p.add_argument('--token_keep_k',      type=int, default=32)
    p.add_argument('--token_keep_k1',     type=int, default=48)
    p.add_argument('--token_keep_k2',     type=int, default=32)
    p.add_argument('--token_keep_k3',     type=int, default=16)
    p.add_argument('--output_csv',        type=str, default=None,
                   help='Save results to CSV (optional)')
    p.add_argument('--models', nargs='+',
                   default=['vanilla', 'tokendrop', 'progressive'],
                   choices=['vanilla', 'tokendrop', 'progressive'])
    args = p.parse_args()

    # GPU memory growth
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    device = 'GPU' if tf.config.list_physical_devices('GPU') else 'CPU'
    print(f'\n{"="*60}')
    print(f'  Latency Benchmark')
    print(f'  Device: {device}')
    print(f'  Config: batch={args.batch_size}, seq={args.seq_len}, '
          f'layers={args.num_layers}, hidden={args.hidden_size}')
    print(f'  Warmup: {args.n_warmup} runs, Measure: {args.n_runs} runs')
    print(f'{"="*60}\n')

    # Build models
    all_models = build_models(args)
    model_dict = {n: m for n, m in all_models}

    # Create dummy batch
    batch = create_dummy_batch(args.batch_size, args.seq_len)

    # Benchmark
    results = {}
    for name in args.models:
        model = model_dict[name]
        print(f'  Benchmarking {name}...')
        mean_ms, std_ms, all_times = measure_latency(
            model, batch, n_warmup=args.n_warmup, n_runs=args.n_runs)
        results[name] = {'mean_ms': mean_ms, 'std_ms': std_ms, 'times': all_times}
        print(f'    {name:<15} {mean_ms:>8.2f} ms  (+/- {std_ms:.2f} ms)')

    # Summary table
    print(f'\n  {"Model":<15} {"Mean (ms)":>10} {"Std (ms)":>10} {"vs Vanilla":>12}')
    print(f'  {"-"*15} {"-"*10} {"-"*10} {"-"*12}')
    van_mean = results.get('vanilla', {}).get('mean_ms', 1)
    for name in args.models:
        r = results[name]
        ratio = r['mean_ms'] / van_mean if van_mean > 0 else 0
        print(f'  {name:<15} {r["mean_ms"]:>10.2f} {r["std_ms"]:>10.2f} {ratio:>11.1%}')

    # Save CSV if requested
    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv) or '.', exist_ok=True)
        with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=[
                'model', 'batch_size', 'seq_len', 'device',
                'mean_ms', 'std_ms', 'n_runs', 'vs_vanilla'])
            w.writeheader()
            for name in args.models:
                r = results[name]
                ratio = r['mean_ms'] / van_mean if van_mean > 0 else 0
                w.writerow({
                    'model': name, 'batch_size': args.batch_size,
                    'seq_len': args.seq_len, 'device': device,
                    'mean_ms': round(r['mean_ms'], 3),
                    'std_ms': round(r['std_ms'], 3),
                    'n_runs': args.n_runs, 'vs_vanilla': round(ratio, 4)})
        print(f'\n  CSV saved: {args.output_csv}')

    print(f'\n{"="*60}\n')
    return results


if __name__ == '__main__':
    main()
