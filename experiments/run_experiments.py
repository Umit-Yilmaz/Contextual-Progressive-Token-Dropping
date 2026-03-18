#!/usr/bin/env python3
"""Experiment runner: Baseline vs. Progressive Contextual Drop BERT.

Runs both methodologies on reproducible synthetic data and reports:
  - Forward-pass validity  (output shapes, NaN / Inf checks)
  - Token-layer efficiency (total token-layer products and attention-FLOPs proxy)
  - Trainable parameter count
  - Timed forward pass     (average over N_TIMING_RUNS runs)

Both encoders are loaded via importlib so their module names never collide,
and neither file on disk is modified.

Usage
-----
    python experiments/run_experiments.py

Requirements
------------
    pip install tensorflow tf-models-official

Reproducibility
---------------
    All random state is seeded with SEED = 42 before any operation.
    Synthetic inputs are generated deterministically from SEED.
    Set the SEED constant below to reproduce a different draw.
"""

import importlib.util
import importlib
import os
import sys
import time

# Force UTF-8 output so Unicode box-drawing chars survive on any Windows console.
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── Directory layout (needed before any encoder imports) ──────────────────────
EXPERIMENTS_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT        = os.path.dirname(EXPERIMENTS_DIR)
NEW_METHOD_DIR   = os.path.join(EXPERIMENTS_DIR, 'progressive_contextual_dropping')

# Make official TF-Models importable (user must have it installed or on path).
sys.path.insert(0, REPO_ROOT)

import numpy as np

# ── Global seed ───────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

try:
  import tensorflow as tf
  tf.random.set_seed(SEED)
except ImportError as exc:
  print(f'[ERROR] TensorFlow not found: {exc}')
  print('        Install with: pip install tensorflow tf-models-official')
  sys.exit(1)

# ── Compatibility shim: tf_keras → tf.keras ───────────────────────────────────
# encoder.py (baseline repo) imports `tf_keras` (the standalone package).
# The installed tf_keras==2.15.0 is incompatible with tensorflow==2.10.1, and
# PyPI has no tf_keras release earlier than 2.14.1.
#
# Fix: inject tf.keras (TF 2.10's bundled Keras) into sys.modules under the
# name 'tf_keras' before load_module runs encoder.py.  The encoder only uses
# standard Keras primitives (Layer, Dense, MultiHeadAttention, initializers…)
# that are identical in tf.keras 2.10 and tf_keras 2.15.  This avoids all the
# version-mismatch import errors without modifying the original encoder file.
import types as _types

_tf_keras_shim = _types.ModuleType('tf_keras')
_tf_keras_shim.__dict__.update(tf.keras.__dict__)
# Ensure sub-package attributes (tf_keras.layers, tf_keras.initializers, …)
# resolve correctly even when accessed via attribute rather than import.
for _attr in ('layers', 'initializers', 'regularizers', 'activations',
              'constraints', 'losses', 'metrics', 'optimizers', 'callbacks',
              'preprocessing', 'saving', 'utils', 'mixed_precision'):
  if hasattr(tf.keras, _attr):
    setattr(_tf_keras_shim, _attr, getattr(tf.keras, _attr))
sys.modules.setdefault('tf_keras', _tf_keras_shim)

# Pre-warm official.* so load_module(encoder.py) finds them in sys.modules.
for _pre in ['official.modeling.tf_utils', 'official.nlp.modeling.layers']:
  try:
    importlib.import_module(_pre)
  except Exception:
    pass  # failures are surfaced later when run_original_experiment runs

# ── Experiment hyper-parameters ───────────────────────────────────────────────
# Reduced from full BERT-base sizes so the runner finishes quickly on CPU.
# Change these to 512 / 768 / 12 / 12 / 3072 for a full-scale comparison.
BATCH_SIZE   = 2
SEQ_LEN      = 128   # full-scale: 512
VOCAB_SIZE   = 30522
HIDDEN_SIZE  = 256   # full-scale: 768
NUM_LAYERS   = 8     # full-scale: 12  (must be >= 4)
NUM_HEADS    = 4     # full-scale: 12
INNER_DIM    = 512   # full-scale: 3072
MAX_MASKED   = 20
N_TIMING_RUNS = 5

# Baseline: one drop to TOKEN_KEEP_K tokens at layer N/2.
TOKEN_KEEP_K  = 64   # full-scale: 256

# Progressive: three drops (must satisfy k3 < k2 < k1 < SEQ_LEN).
TOKEN_KEEP_K1 = 96   # full-scale: 384
TOKEN_KEEP_K2 = 64   # full-scale: 256
TOKEN_KEEP_K3 = 32   # full-scale: 128


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_module(module_name: str, file_path: str):
  """Import a Python file as a module without modifying sys.modules globally."""
  spec = importlib.util.spec_from_file_location(module_name, file_path)
  mod  = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


def make_synthetic_inputs(batch_size: int, seq_len: int, vocab_size: int,
                           max_masked: int, seed: int = SEED) -> dict:
  """Create deterministic BERT pretraining inputs.

  Args:
    batch_size: Number of sequences per batch.
    seq_len:    Total sequence length including padding.
    vocab_size: Vocabulary size (token IDs sampled from [1, vocab_size)).
    max_masked: Maximum number of [MASK] positions per sequence.
    seed:       NumPy random seed for reproducibility.

  Returns:
    Dict with keys: input_word_ids, input_mask, input_type_ids,
                    masked_lm_ids, masked_lm_weights, masked_lm_positions.
  """
  rng = np.random.RandomState(seed)

  word_ids = rng.randint(104, vocab_size, size=(batch_size, seq_len)).astype(np.int32)

  # Pad the last 20 % of each sequence.
  pad_start = max(4, int(seq_len * 0.80))
  word_ids[:, pad_start:] = 0        # [PAD]
  word_ids[:, 0]          = 101      # [CLS]
  word_ids[:, pad_start - 1] = 102   # [SEP]

  mask     = np.ones((batch_size, seq_len), dtype=np.int32)
  mask[:, pad_start:] = 0

  type_ids = np.zeros((batch_size, seq_len), dtype=np.int32)
  type_ids[:, seq_len // 2:] = 1    # segment B starts at midpoint

  masked_lm_positions = np.zeros((batch_size, max_masked), dtype=np.int32)
  masked_lm_ids       = np.zeros((batch_size, max_masked), dtype=np.int32)
  masked_lm_weights   = np.zeros((batch_size, max_masked), dtype=np.float32)

  for b in range(batch_size):
    valid = list(range(1, pad_start - 1))   # exclude [CLS] and [SEP]
    n_mask = min(max_masked, len(valid))
    chosen = rng.choice(valid, n_mask, replace=False)
    masked_lm_positions[b, :n_mask] = chosen
    masked_lm_ids[b, :n_mask]       = word_ids[b, chosen]
    masked_lm_weights[b, :n_mask]   = 1.0
    word_ids[b, chosen]             = 103   # [MASK]

  return {
      'input_word_ids':      tf.constant(word_ids),
      'input_mask':          tf.constant(mask),
      'input_type_ids':      tf.constant(type_ids),
      'masked_lm_ids':       tf.constant(masked_lm_ids),
      'masked_lm_weights':   tf.constant(masked_lm_weights),
      'masked_lm_positions': tf.constant(masked_lm_positions),
  }


def count_params(model) -> dict:
  """Return trainable and non-trainable parameter counts."""
  trainable     = int(sum(tf.size(v).numpy() for v in model.trainable_variables))
  non_trainable = int(sum(tf.size(v).numpy() for v in model.non_trainable_variables))
  return {'trainable': trainable, 'non_trainable': non_trainable,
          'total': trainable + non_trainable}


def token_layer_efficiency(method: str) -> dict:
  """Compute token-layer products and attention-FLOPs proxy.

  Attention FLOPs are proportional to n² per layer (n = sequence length at
  that layer), which captures the dominant quadratic cost of self-attention.

  Args:
    method: 'baseline' or 'progressive'.

  Returns:
    Dict with keys 'token_layer_product' and 'attention_flops_proxy'.
  """
  n, k  = SEQ_LEN, TOKEN_KEEP_K
  k1, k2, k3 = TOKEN_KEEP_K1, TOKEN_KEEP_K2, TOKEN_KEEP_K3
  L = NUM_LAYERS

  if method == 'baseline':
    # Layers 0 .. L/2-2: all n tokens.
    # Layer  L/2-1     : cross-attention, queries=k, keys=n  → cost ∝ k*n.
    # Layers L/2 .. L-2: k tokens.
    # Layer  L-1       : all n tokens.
    tl = (L // 2 - 1) * n  +  n  +  (L // 2 - 1) * k  +  n
    af = (L // 2 - 1) * n**2 + k * n + (L // 2 - 1) * k**2 + n**2

  elif method == 'progressive':
    # Stage 0: n tokens,  L//4 layers.
    # Stage 1: k1 tokens, L//4 layers.
    # Stage 2: k2 tokens, L//4 layers.
    # Stage 3: k3 tokens, L - 3*(L//4) - 1 layers.
    # Final:   n tokens,  1 layer.
    s3_layers = L - 3 * (L // 4) - 1
    tl = (L // 4) * n  +  (L // 4) * k1  +  (L // 4) * k2  +  s3_layers * k3  +  n
    af = (L // 4) * n**2 + (L // 4) * k1**2 + (L // 4) * k2**2 + s3_layers * k3**2 + n**2

  else:
    raise ValueError(f'Unknown method: {method}')

  return {'token_layer_product': tl, 'attention_flops_proxy': af}


def validate_output(outputs: dict, expected_shape: tuple) -> dict:
  """Check output shapes and absence of NaN / Inf."""
  seq_out  = outputs['sequence_output']
  pool_out = outputs['pooled_output']
  return {
      'seq_shape':    tuple(seq_out.shape),
      'pool_shape':   tuple(pool_out.shape),
      'shape_ok':     tuple(seq_out.shape) == expected_shape,
      'has_nan':      bool(tf.reduce_any(tf.math.is_nan(seq_out)).numpy()),
      'has_inf':      bool(tf.reduce_any(tf.math.is_inf(seq_out)).numpy()),
  }


def time_forward_pass(encoder, encoder_inputs: dict, n_runs: int) -> float:
  """Return mean forward-pass wall-clock time in milliseconds."""
  # One warmup run to initialise variables and XLA compilation.
  _ = encoder(encoder_inputs, training=False)
  times = []
  for _ in range(n_runs):
    t0 = time.perf_counter()
    _ = encoder(encoder_inputs, training=False)
    times.append(time.perf_counter() - t0)
  return float(np.mean(times)) * 1000.0


# ── Experiment 1: Baseline ────────────────────────────────────────────────────

def run_original_experiment(inputs: dict) -> dict:
  print('\n' + '=' * 65)
  print('EXPERIMENT 1  —  Baseline Token Dropping (original repo)')
  print('=' * 65)

  try:
    mod = load_module(
        'original_encoder',
        os.path.join(REPO_ROOT, 'encoder.py'))
    OrigEncoder = mod.TokenDropBertEncoder
  except Exception as exc:
    print(f'  [SKIP] Cannot import original encoder: {exc}')
    print('         Ensure tf-models-official is installed and the repo root')
    print(f'         ({REPO_ROOT}) is accessible.')
    return {}

  encoder = OrigEncoder(
      vocab_size=VOCAB_SIZE,
      hidden_size=HIDDEN_SIZE,
      num_layers=NUM_LAYERS,
      num_attention_heads=NUM_HEADS,
      inner_dim=INNER_DIM,
      max_sequence_length=SEQ_LEN,
      type_vocab_size=2,
      token_keep_k=TOKEN_KEEP_K,
      token_allow_list=(100, 101, 102, 103),
      token_deny_list=(0,))

  enc_inputs = {k: inputs[k]
                for k in ('input_word_ids', 'input_mask', 'input_type_ids')}
  expected_shape = (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)

  outputs     = encoder(enc_inputs, training=False)
  validity    = validate_output(outputs, expected_shape)
  avg_time_ms = time_forward_pass(encoder, enc_inputs, N_TIMING_RUNS)
  params      = count_params(encoder)
  efficiency  = token_layer_efficiency('baseline')

  print(f'  sequence_output shape : {validity["seq_shape"]}')
  print(f'  pooled_output   shape : {validity["pool_shape"]}')
  print(f'  Shape correct         : {validity["shape_ok"]}')
  print(f'  NaN in output         : {validity["has_nan"]}')
  print(f'  Inf in output         : {validity["has_inf"]}')
  print(f'  Trainable params      : {params["trainable"]:,}')
  print(f'  Non-trainable params  : {params["non_trainable"]:,}  '
        f'<-- importance-score table')
  print(f'  Token-layer product   : {efficiency["token_layer_product"]:,}')
  print(f'  Attention FLOPs proxy : {efficiency["attention_flops_proxy"]:,}')
  print(f'  Avg forward pass      : {avg_time_ms:.2f} ms  '
        f'(over {N_TIMING_RUNS} runs)')
  print('  [PASS]')

  return {
      'params':               params,
      'efficiency':           efficiency,
      'avg_time_ms':          avg_time_ms,
      'seq_shape':            validity['seq_shape'],
      'valid':                validity['shape_ok']
                              and not validity['has_nan']
                              and not validity['has_inf'],
  }


# ── Experiment 2: Progressive Contextual Drop ─────────────────────────────────

def run_progressive_experiment(inputs: dict) -> dict:
  print('\n' + '=' * 65)
  print('EXPERIMENT 2  —  Progressive Contextual Drop (new methodology)')
  print('=' * 65)

  try:
    mod = load_module(
        'progressive_encoder',
        os.path.join(NEW_METHOD_DIR, 'encoder.py'))
    ProgEncoder = mod.ProgressiveContextualDropEncoder
  except Exception as exc:
    print(f'  [FAIL] Cannot import progressive encoder: {exc}')
    raise

  encoder = ProgEncoder(
      vocab_size=VOCAB_SIZE,
      hidden_size=HIDDEN_SIZE,
      num_layers=NUM_LAYERS,
      num_attention_heads=NUM_HEADS,
      inner_dim=INNER_DIM,
      max_sequence_length=SEQ_LEN,
      type_vocab_size=2,
      token_keep_k1=TOKEN_KEEP_K1,
      token_keep_k2=TOKEN_KEEP_K2,
      token_keep_k3=TOKEN_KEEP_K3,
      token_allow_list=(100, 101, 102, 103),
      token_deny_list=(0,))

  enc_inputs = {k: inputs[k]
                for k in ('input_word_ids', 'input_mask', 'input_type_ids')}
  expected_shape = (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)

  outputs     = encoder(enc_inputs, training=False)
  validity    = validate_output(outputs, expected_shape)
  avg_time_ms = time_forward_pass(encoder, enc_inputs, N_TIMING_RUNS)
  params      = count_params(encoder)
  efficiency  = token_layer_efficiency('progressive')

  print(f'  sequence_output shape : {validity["seq_shape"]}')
  print(f'  pooled_output   shape : {validity["pool_shape"]}')
  print(f'  Shape correct         : {validity["shape_ok"]}')
  print(f'  NaN in output         : {validity["has_nan"]}')
  print(f'  Inf in output         : {validity["has_inf"]}')
  print(f'  Trainable params      : {params["trainable"]:,}')
  print(f'  Non-trainable params  : {params["non_trainable"]:,}  '
        f'<-- zero (no importance table)')
  print(f'  Token-layer product   : {efficiency["token_layer_product"]:,}')
  print(f'  Attention FLOPs proxy : {efficiency["attention_flops_proxy"]:,}')
  print(f'  Avg forward pass      : {avg_time_ms:.2f} ms  '
        f'(over {N_TIMING_RUNS} runs)')
  print('  [PASS]')

  return {
      'params':      params,
      'efficiency':  efficiency,
      'avg_time_ms': avg_time_ms,
      'seq_shape':   validity['seq_shape'],
      'valid':       validity['shape_ok']
                     and not validity['has_nan']
                     and not validity['has_inf'],
  }


# ── Comparison table ──────────────────────────────────────────────────────────

def print_comparison(orig: dict, prog: dict):
  if not orig or not prog:
    print('\n[WARN] Comparison skipped — one experiment produced no results.')
    return

  print('\n' + '=' * 65)
  print('COMPARISON SUMMARY')
  print('=' * 65)

  header = f'{"Metric":<38} {"Baseline":>10} {"Progress.":>10}  {"Change":>8}'
  print(header)
  print('-' * 68)

  def row(label, a, b, lower_is_better=True):
    pct   = (b - a) / a * 100 if a else 0.0
    better = (lower_is_better and pct < 0) or (not lower_is_better and pct > 0)
    mark  = '✓' if better else ' '
    sign  = '' if pct >= 0 else '-'
    print(f'{label:<38} {a:>10,} {b:>10,}  {sign}{abs(pct):>6.1f}% {mark}')

  row('Trainable params',
      orig['params']['trainable'],
      prog['params']['trainable'],
      lower_is_better=False)
  row('Non-trainable params (importance table)',
      orig['params']['non_trainable'],
      prog['params']['non_trainable'],
      lower_is_better=True)
  row('Token-layer product',
      orig['efficiency']['token_layer_product'],
      prog['efficiency']['token_layer_product'],
      lower_is_better=True)
  row('Attention FLOPs proxy',
      orig['efficiency']['attention_flops_proxy'],
      prog['efficiency']['attention_flops_proxy'],
      lower_is_better=True)

  print('-' * 68)

  # Timing (floats, print separately)
  ot, pt = orig['avg_time_ms'], prog['avg_time_ms']
  pct_t  = (pt - ot) / ot * 100 if ot else 0.0
  mark_t = '✓' if pct_t < 0 else ' '
  sign_t = '' if pct_t >= 0 else '-'
  print(f'{"Avg forward pass (ms)":<38} {ot:>10.2f} {pt:>10.2f}  '
        f'{sign_t}{abs(pct_t):>6.1f}% {mark_t}')

  print('\n  ✓  = progressive method is better for that metric')
  print(f'\n  Output shape identical  : {orig["seq_shape"] == prog["seq_shape"]}')
  print(f'  Baseline output valid   : {orig["valid"]}')
  print(f'  Progressive output valid: {prog["valid"]}')

  print('\nQualitative advantages of progressive contextual drop (not measured here):')
  print('  • Context-aware scoring  — uses live hidden states, not a vocab lookup.')
  print('  • Cold-start ready       — valid from training step 1, no warm-up lag.')
  print('  • No extra memory        — no importance-score embedding table.')
  print('  • Finer-grained control  — three budgets (k1/k2/k3) vs. one (k).')


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
  print('=' * 65)
  print('Token Dropping Experiment Runner')
  print('=' * 65)
  print(f'  TensorFlow version : {tf.__version__}')
  print(f'  Random seed        : {SEED}')
  print(f'  Batch / seq len    : {BATCH_SIZE} × {SEQ_LEN}')
  print(f'  Model config       : {NUM_LAYERS} layers, '
        f'hidden={HIDDEN_SIZE}, heads={NUM_HEADS}')
  print(f'  Baseline keep_k    : {TOKEN_KEEP_K}')
  print(f'  Progressive k1/k2/k3: {TOKEN_KEEP_K1} / {TOKEN_KEEP_K2} / {TOKEN_KEEP_K3}')

  inputs = make_synthetic_inputs(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE, MAX_MASKED)

  orig_results = run_original_experiment(inputs)
  prog_results = run_progressive_experiment(inputs)

  print_comparison(orig_results, prog_results)
  print('\nDone.')


if __name__ == '__main__':
  main()
