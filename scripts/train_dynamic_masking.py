#!/usr/bin/env python3
# =============================================================================
# train_dynamic_masking.py  —  3-Way BERT Comparison (Dynamic MLM Masking)
#
# Same architecture and training setup as train_csv_comparison.py, but with
# DYNAMIC masking: each epoch applies fresh random MLM masks to the raw
# (unmasked) token IDs, shared across all three models for fair comparison.
#
# Key design: EPOCH-BASED loop (not model-based). Each epoch:
#   1. Apply MLM masking to raw data with a deterministic seed (epoch number)
#   2. Feed the SAME masked data to all three models
#   3. Each model trains on identical inputs → fair comparison
#
# Requires unmasked CSV data (created by prepare_unmasked_data.py):
#   input_ids      – original token IDs (no [MASK] tokens)
#   attention_mask – 1 for real tokens, 0 for padding
#
# Usage:
#   python scripts/train_dynamic_masking.py \
#     --data_path /data/umityilmaz/token_drop_v2/data/wikitext_unmasked_512.csv \
#     --output_dir /data/umityilmaz/token_drop_v2/checkpoints/run4_dynamic \
#     --epochs 50000 --max_steps 200000 --batch_size 16 --learning_rate 1e-4 \
#     --weight_decay 0.01 --warmup_ratio 0.06 \
#     --early_stopping_patience 10 --log_every 500 \
#     --hidden_size 768 --num_layers 12 --num_heads 12 --intermediate_size 3072 \
#     --max_seq_len 512 --token_keep_k 256 \
#     --token_keep_k1 384 --token_keep_k2 256 --token_keep_k3 128 \
#     --models vanilla tokendrop progressive
#
# =============================================================================

import os
import sys
import csv
import json
import time
import math
import logging
import datetime
import argparse
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*lambda.*')

# ── Repo path ─────────────────────────────────────────────────────────────────
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'experiments', 'progressive_contextual_dropping'))
sys.path.insert(0, REPO)   # highest priority: repo root

import numpy as np
import tensorflow as tf

# ── ANSI colours ──────────────────────────────────────────────────────────────
class C:
    BOLD    = '\033[1m'
    RESET   = '\033[0m'
    BLUE    = '\033[94m'
    YELLOW  = '\033[93m'
    GREEN   = '\033[92m'
    RED     = '\033[91m'
    CYAN    = '\033[96m'
    MAGENTA = '\033[95m'

VOCAB_SIZE = 30522   # bert-base-uncased vocab

# ── MLM masking constants ─────────────────────────────────────────────────────
MASK_TOKEN_ID = 103
CLS_TOKEN_ID  = 101
SEP_TOKEN_ID  = 102
PAD_TOKEN_ID  = 0
UNK_TOKEN_ID  = 100

_SPECIAL = {CLS_TOKEN_ID, SEP_TOKEN_ID, PAD_TOKEN_ID, UNK_TOKEN_ID}

MASK_PROB      = 0.15
MASK_REPLACE   = 0.80
RANDOM_REPLACE = 0.10


# ── Deterministic numpy MLM masking ──────────────────────────────────────────

def apply_epoch_masking(input_ids, attention_mask, seed):
    """Apply MLM masking to entire dataset using numpy with a fixed seed.

    All models share the same masked output for fair comparison.

    Args:
        input_ids:      [N, seq_len] np.int32 — original (unmasked) token IDs
        attention_mask:  [N, seq_len] np.int32 — 1=real, 0=pad
        seed:           int — deterministic seed (different per epoch)

    Returns:
        masked_ids: [N, seq_len] np.int32
        labels:     [N, seq_len] np.int32 (-100 for non-masked)
    """
    rng = np.random.RandomState(seed)
    n, seq_len = input_ids.shape

    masked_ids = input_ids.copy()
    labels = np.full_like(input_ids, -100)

    # Vectorized: identify eligible positions (real tokens, not special)
    is_special = np.isin(input_ids, list(_SPECIAL))
    is_eligible = (attention_mask == 1) & (~is_special)

    # Random selection: 15% of eligible tokens
    rand_select = rng.random((n, seq_len)).astype(np.float32)
    is_selected = is_eligible & (rand_select < MASK_PROB)

    # Store original IDs as labels at selected positions
    labels[is_selected] = input_ids[is_selected]

    # Replacement strategy
    rand_replace = rng.random((n, seq_len)).astype(np.float32)

    # 80% -> [MASK]
    use_mask = is_selected & (rand_replace < MASK_REPLACE)
    masked_ids[use_mask] = MASK_TOKEN_ID

    # 10% -> random token (avoid special tokens: range 999..vocab_size-1)
    use_random = is_selected & (rand_replace >= MASK_REPLACE) & \
                 (rand_replace < MASK_REPLACE + RANDOM_REPLACE)
    n_random = use_random.sum()
    if n_random > 0:
        masked_ids[use_random] = rng.randint(999, VOCAB_SIZE, size=n_random)

    # 10% -> keep original (implicit)

    return masked_ids, labels


# ── LR Schedule: linear warmup -> linear decay to 0 ──────────────────────────

class WarmupLinearDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, peak_lr, warmup_steps, total_steps):
        super().__init__()
        self.peak_lr      = float(peak_lr)
        self.warmup_steps = float(max(warmup_steps, 1))
        self.total_steps  = float(max(total_steps, warmup_steps + 1))

    def __call__(self, step):
        step       = tf.cast(step, tf.float32)
        warmup_lr  = self.peak_lr * step / self.warmup_steps
        decay_steps = self.total_steps - self.warmup_steps
        decay_lr   = self.peak_lr * tf.maximum(
            0.0, (self.total_steps - step) / tf.maximum(decay_steps, 1.0))
        return tf.where(step < self.warmup_steps, warmup_lr, decay_lr)

    def get_config(self):
        return {'peak_lr': self.peak_lr, 'warmup_steps': self.warmup_steps,
                'total_steps': self.total_steps}


# ── Early Stopping ────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best      = float('inf')

    def step(self, val_loss):
        if val_loss < self.best - self.min_delta:
            self.best    = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


# ── Logging helpers ───────────────────────────────────────────────────────────

def setup_logger(name, log_file):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s',
                             datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def log_epoch(logger, name, epoch, n_epochs, t_loss, t_acc, v_loss, v_acc, sps,
              lr=None):
    t_ppl = math.exp(min(t_loss, 20))
    v_ppl = math.exp(min(v_loss, 20))
    lr_str = f" | LR: {lr:.3e}" if lr is not None else ""
    logger.info(
        f"Epoch {epoch}/{n_epochs}{lr_str} | "
        f"Train - Loss: {t_loss:.4f}, Acc: {t_acc:.4f}, "
        f"MLM Acc: {t_acc:.4f}, PPL: {t_ppl:.2f} | "
        f"Val - Loss: {v_loss:.4f}, Acc: {v_acc:.4f}, "
        f"MLM Acc: {v_acc:.4f}, PPL: {v_ppl:.2f} | "
        f"Steps/s: {sps:.2f}")


def append_epoch_csv(csv_path, row):
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def save_step_log(log_path, model_name, global_step, loss, acc):
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"{ts} | INFO | [{model_name}] step={global_step} "
                f"loss={float(loss):.4f} acc={float(acc):.4f} "
                f"ppl={math.exp(min(float(loss), 20)):.2f}\n")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_unmasked_csv(csv_path, max_samples=None, val_ratio=0.05, test_ratio=0.05,
                      seed=42):
    """Load unmasked CSV into numpy arrays.

    Returns:
      (train_data, val_data, test_data) each a dict with keys
      'input_ids', 'attention_mask' (np.int32 arrays).
    """
    print(f"\n{C.CYAN}Loading unmasked CSV data:{C.RESET} {csv_path}")
    ids_list, mask_list = [], []

    with open(csv_path, 'r', encoding='utf-8') as fh:
        reader = csv.reader(fh)
        next(reader)
        for i, row in enumerate(reader):
            if max_samples and i >= max_samples:
                break
            if len(row) < 2:
                continue
            ids_list.append(json.loads(row[0]))
            mask_list.append(json.loads(row[1]))

    input_ids      = np.array(ids_list,  dtype=np.int32)
    attention_mask = np.array(mask_list, dtype=np.int32)
    n, seq_len     = input_ids.shape

    # Shuffle + 3-way split
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test  = max(1, int(n * test_ratio))
    n_val   = max(1, int(n * val_ratio))
    n_train = n - n_val - n_test

    idx_train = perm[:n_train]
    idx_val   = perm[n_train:n_train + n_val]
    idx_test  = perm[n_train + n_val:]

    train_data = dict(input_ids=input_ids[idx_train],
                      attention_mask=attention_mask[idx_train])
    val_data   = dict(input_ids=input_ids[idx_val],
                      attention_mask=attention_mask[idx_val])
    test_data  = dict(input_ids=input_ids[idx_test],
                      attention_mask=attention_mask[idx_test])

    real_per_sample = attention_mask.sum(axis=1).mean()
    expected_masked = real_per_sample * MASK_PROB
    print(f"  Samples   : {n:,}  (train={n_train:,}, val={n_val:,}, test={n_test:,})")
    print(f"  Seq len   : {seq_len}")
    print(f"  Real tok/s: ~{real_per_sample:.0f}")
    print(f"  Expected masked/s: ~{expected_masked:.1f} ({MASK_PROB:.0%} dynamic)")
    print(f"  {C.MAGENTA}Masking: DYNAMIC (applied fresh each epoch, shared across models){C.RESET}")
    return train_data, val_data, test_data


def make_masked_dataset(input_ids, attention_mask, labels, batch_size,
                        shuffle=True, seed=0):
    """Create a tf.data.Dataset from pre-masked numpy arrays."""
    data = {
        'input_ids':      input_ids,
        'attention_mask': attention_mask,
        'labels':         labels,
    }
    ds = tf.data.Dataset.from_tensor_slices(data)
    if shuffle:
        ds = ds.shuffle(
            buffer_size=min(10_000, len(input_ids)),
            seed=seed,
            reshuffle_each_iteration=False)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def make_static_eval_dataset(data, batch_size, seed=42):
    """Create eval dataset with FIXED masks (deterministic for fair comparison)."""
    masked_ids, labels = apply_epoch_masking(
        data['input_ids'], data['attention_mask'], seed=seed)
    return make_masked_dataset(
        masked_ids, data['attention_mask'], labels,
        batch_size, shuffle=False)


# ── Loss & accuracy ──────────────────────────────────────────────────────────

def mlm_loss_acc(logits, labels):
    mask     = tf.cast(tf.not_equal(labels, -100), tf.float32)
    safe_lbl = tf.where(labels == -100, tf.zeros_like(labels), labels)
    per_tok  = tf.keras.losses.sparse_categorical_crossentropy(
        safe_lbl, tf.cast(logits, tf.float32), from_logits=True)
    n    = tf.reduce_sum(mask) + 1e-9
    loss = tf.reduce_sum(per_tok * mask) / n
    preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
    acc   = tf.reduce_sum(tf.cast(tf.equal(preds, safe_lbl), tf.float32) * mask) / n
    return loss, acc, per_tok


# ── BERT MLM model wrapper ───────────────────────────────────────────────────

class BertMLM(tf.keras.Model):
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


# ── Model factory ─────────────────────────────────────────────────────────────

def build_models(args):
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
        vocab_size          = VOCAB_SIZE,
        inner_dim           = args.intermediate_size,
        max_sequence_length = args.max_seq_len,
        dict_outputs        = True,
        **common)
    vanilla_model = BertMLM(vanilla_enc, VOCAB_SIZE, args.hidden_size,
                             name='vanilla_mlm')

    tokendrop_enc = TokenDropBertEncoder(
        vocab_size          = VOCAB_SIZE,
        intermediate_size   = args.intermediate_size,
        max_sequence_length = args.max_seq_len,
        token_keep_k        = args.token_keep_k,
        **common)
    tokendrop_model = BertMLM(tokendrop_enc, VOCAB_SIZE, args.hidden_size,
                               name='tokendrop_mlm')

    prog_enc = ProgressiveContextualDropEncoder(
        vocab_size          = VOCAB_SIZE,
        inner_dim           = args.intermediate_size,
        max_sequence_length = args.max_seq_len,
        token_keep_k1       = args.token_keep_k1,
        token_keep_k2       = args.token_keep_k2,
        token_keep_k3       = args.token_keep_k3,
        **common)
    prog_model = BertMLM(prog_enc, VOCAB_SIZE, args.hidden_size,
                          name='progressive_mlm')

    return [
        ('vanilla',     vanilla_model,   C.BLUE),
        ('tokendrop',   tokendrop_model, C.YELLOW),
        ('progressive', prog_model,      C.GREEN),
    ]


# ── Training / eval steps ────────────────────────────────────────────────────

@tf.function
def train_step(model, optimizer, batch):
    with tf.GradientTape() as tape:
        logits                  = model(batch, training=True)
        loss, acc, per_tok_loss = mlm_loss_acc(logits, batch['labels'])
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    enc = model.encoder
    if hasattr(enc, 'record_mlm_loss'):
        mlm_mask = tf.cast(tf.not_equal(batch['labels'], -100), tf.float32)
        true_ids = tf.where(
            batch['labels'] == -100, batch['input_ids'], batch['labels'])
        enc.record_mlm_loss(mlm_ids=true_ids, mlm_losses=per_tok_loss * mlm_mask)

    return loss, acc


@tf.function
def eval_step(model, batch):
    logits       = model(batch, training=False)
    loss, acc, _ = mlm_loss_acc(logits, batch['labels'])
    return loss, acc


# ── Final comparison table ────────────────────────────────────────────────────

def print_comparison(results, threshold=0.10, speed_threshold=0.85):
    print(f"\n{C.BOLD}{'=' * 70}{C.RESET}")
    print(f"{C.BOLD}  3-Way Comparison — Dynamic Masking — Final Results{C.RESET}")
    print(f"{C.BOLD}{'=' * 70}{C.RESET}")
    header = (f"  {'Model':<22} {'Val Loss':>9} {'Val Acc':>8} "
              f"{'Test Loss':>10} {'Test Acc':>9} "
              f"{'Train Loss':>10} {'Best Ep':>7} {'Steps/s':>8}")
    print(header)
    print(f"  {'-'*22} {'-'*9} {'-'*8} {'-'*10} {'-'*9} {'-'*10} {'-'*7} {'-'*8}")

    color_map = {'vanilla': C.BLUE, 'tokendrop': C.YELLOW, 'progressive': C.GREEN}
    for name, r in results.items():
        c = color_map.get(name, '')
        print(f"  {c}{name:<22}{C.RESET}"
              f" {r['val_loss']:>9.4f}"
              f" {r['val_acc']:>8.4f}"
              f" {r.get('test_loss', 0):>10.4f}"
              f" {r.get('test_acc', 0):>9.4f}"
              f" {r['train_loss']:>10.4f}"
              f" {r.get('best_epoch', '-'):>7}"
              f" {r.get('steps_per_second', 0):>8.1f}")

    print()
    if 'vanilla' in results and 'progressive' in results:
        ref  = results['vanilla']
        prog = results['progressive']
        loss_ratio  = prog['val_loss'] / max(ref['val_loss'],  1e-9)
        speed_ratio = (prog.get('steps_per_second', 1) /
                       max(ref.get('steps_per_second', 1), 1e-9))

        loss_ok  = loss_ratio  <= (1 + threshold)
        speed_ok = speed_ratio >= speed_threshold

        def _chk(ok, label, detail):
            sym = f"{C.GREEN}V{C.RESET}" if ok else f"{C.RED}X{C.RESET}"
            sta = f"{C.GREEN}GO{C.RESET}" if ok else f"{C.RED}NO-GO{C.RESET}"
            print(f"  {sym}  {label:<50} [{sta}]")
            print(f"      -> {detail}")

        _chk(loss_ok,
             f"Progressive val_loss <= vanilla x {1+threshold:.2f}",
             f"progressive={prog['val_loss']:.4f}, "
             f"vanilla={ref['val_loss']:.4f}, ratio={loss_ratio:.3f}")
        _chk(speed_ok,
             f"Progressive speed >= vanilla x {speed_threshold:.2f}",
             f"progressive={prog.get('steps_per_second',0):.2f} s/s, "
             f"vanilla={ref.get('steps_per_second',0):.2f} s/s, "
             f"ratio={speed_ratio:.3f}")

        overall_go = loss_ok and speed_ok
        verdict = (
            f"{C.GREEN}{C.BOLD}GO - Progressive drop is viable (dynamic masking){C.RESET}"
            if overall_go else
            f"{C.RED}{C.BOLD}NO-GO - Check pivot strategies in PLAN.md{C.RESET}")
        print(f"\n  {verdict}")

    print(f"{C.BOLD}{'=' * 70}{C.RESET}")


def save_results_json(results, out_dir, args):
    payload = {
        'timestamp': datetime.datetime.now().isoformat(),
        'masking': 'dynamic',
        'shared_masks': True,
        'args': vars(args),
        'results': results,
    }
    path = os.path.join(out_dir, 'results_summary.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    print(f"\n  {C.CYAN}Results saved:{C.RESET} {path}")


def save_results_csv(results, out_dir):
    path = os.path.join(out_dir, 'results_summary.csv')
    fields = ['model', 'val_loss', 'val_acc', 'val_ppl',
              'test_loss', 'test_acc', 'test_ppl',
              'train_loss', 'train_acc', 'best_val_loss', 'best_epoch',
              'steps_per_second', 'forward_latency_ms', 'global_step']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for name, r in results.items():
            row = {'model': name, **r}
            w.writerow(row)
    print(f"  {C.CYAN}CSV summary:{C.RESET}  {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='3-Way BERT Comparison with DYNAMIC MLM Masking (shared masks)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('--data_path', default=
        '/data/umityilmaz/token_drop_v2/data/wikitext_unmasked_512.csv')
    p.add_argument('--output_dir', default=
        '/data/umityilmaz/token_drop_v2/checkpoints/run4_dynamic')
    p.add_argument('--max_samples', type=int, default=0)
    p.add_argument('--val_ratio',   type=float, default=0.05)
    p.add_argument('--test_ratio',  type=float, default=0.05)

    p.add_argument('--epochs',        type=int,   default=5)
    p.add_argument('--max_steps',     type=int,   default=0)
    p.add_argument('--batch_size',    type=int,   default=16)
    p.add_argument('--learning_rate', type=float, default=1e-4)
    p.add_argument('--log_every',     type=int,   default=500)

    p.add_argument('--weight_decay',            type=float, default=0.01)
    p.add_argument('--warmup_ratio',            type=float, default=0.06)
    p.add_argument('--early_stopping_patience', type=int,   default=10)

    p.add_argument('--hidden_size',       type=int,   default=768)
    p.add_argument('--num_layers',        type=int,   default=12)
    p.add_argument('--num_heads',         type=int,   default=12)
    p.add_argument('--intermediate_size', type=int,   default=3072)
    p.add_argument('--max_seq_len',       type=int,   default=512)
    p.add_argument('--dropout_rate',      type=float, default=0.1)

    p.add_argument('--token_keep_k',  type=int, default=256)
    p.add_argument('--token_keep_k1', type=int, default=384)
    p.add_argument('--token_keep_k2', type=int, default=256)
    p.add_argument('--token_keep_k3', type=int, default=128)

    p.add_argument('--models', nargs='+',
                   default=['vanilla', 'tokendrop', 'progressive'],
                   choices=['vanilla', 'tokendrop', 'progressive'])
    p.add_argument('--go_threshold',    type=float, default=0.10)
    p.add_argument('--speed_threshold', type=float, default=0.85)
    return p.parse_args()


def main():
    args = parse_args()

    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    os.makedirs(args.output_dir, exist_ok=True)

    run_log = os.path.join(args.output_dir, 'run.log')
    run_logger = setup_logger('run', run_log)
    run_logger.info(
        f"Run started (DYNAMIC MASKING, shared masks) | models={args.models} | "
        f"epochs={args.epochs} | batch={args.batch_size} | "
        f"lr={args.learning_rate} | hidden={args.hidden_size} | "
        f"layers={args.num_layers} | k_td={args.token_keep_k} | "
        f"k1={args.token_keep_k1} | k2={args.token_keep_k2} | "
        f"k3={args.token_keep_k3}")

    print(f"\n{C.CYAN}{C.BOLD}{'-'*65}")
    print(f"  3-Way BERT CSV Comparison — DYNAMIC MASKING (shared)")
    print(f"  Vanilla  ·  TokenDrop  ·  Progressive Drop")
    print(f"{'-'*65}{C.RESET}")
    print(f"  Data   : {args.data_path}")
    print(f"  Output : {args.output_dir}")
    print(f"  Models : {', '.join(args.models)}")
    print(f"  Config : hidden={args.hidden_size}, layers={args.num_layers}, "
          f"heads={args.num_heads}, intermediate={args.intermediate_size}")
    print(f"  Budget : k(td)={args.token_keep_k}, "
          f"k1={args.token_keep_k1}, k2={args.token_keep_k2}, k3={args.token_keep_k3}")
    print(f"  Train  : epochs={args.epochs}, batch={args.batch_size}, "
          f"lr={args.learning_rate}, weight_decay={args.weight_decay}, "
          f"warmup={args.warmup_ratio:.0%}, "
          f"early_stop_patience={args.early_stopping_patience}")
    print(f"  {C.MAGENTA}{C.BOLD}Masking : DYNAMIC (fresh per epoch, SHARED across models){C.RESET}")

    # ── Load unmasked data ────────────────────────────────────────────────────
    train_data, val_data, test_data = load_unmasked_csv(
        args.data_path,
        max_samples=args.max_samples or None,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio)

    # ── Val and test: fixed masks (deterministic) ─────────────────────────────
    print(f"\n{C.CYAN}Preparing val/test datasets (fixed masks)...{C.RESET}")
    val_ds  = make_static_eval_dataset(val_data,  args.batch_size, seed=42)
    test_ds = make_static_eval_dataset(test_data, args.batch_size, seed=43)
    print(f"  Val/test masks: FIXED (seeded, deterministic)")

    # ── LR schedule parameters ────────────────────────────────────────────────
    n_train         = len(train_data['input_ids'])
    steps_per_epoch = n_train // args.batch_size
    total_steps     = args.epochs * steps_per_epoch
    if args.max_steps > 0:
        total_steps = min(args.max_steps, total_steps)
    warmup_steps    = int(total_steps * args.warmup_ratio)
    print(f"  LR sched: total_steps={total_steps:,}  "
          f"warmup_steps={warmup_steps:,}  ({args.warmup_ratio:.0%})")

    # ── Build models ──────────────────────────────────────────────────────────
    print(f"\n{C.BOLD}Building models...{C.RESET}")
    all_models = build_models(args)
    model_list = [(n, m, c) for n, m, c in all_models if n in args.models]

    # ── Per-model state: optimizer, early stopping, loggers, etc. ─────────────
    model_states = {}
    for name, model, color in model_list:
        log_dir = os.path.join(args.output_dir, name)
        os.makedirs(log_dir, exist_ok=True)

        lr_schedule = WarmupLinearDecay(
            peak_lr=args.learning_rate,
            warmup_steps=warmup_steps,
            total_steps=max(total_steps, 1))
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=args.weight_decay,
            beta_1=0.9, beta_2=0.999, epsilon=1e-8)

        epoch_log_path = os.path.join(log_dir, f'training-{name}.log')
        epoch_csv_path = os.path.join(log_dir, f'training-{name}.csv')
        step_log_path  = os.path.join(log_dir, f'steps-{name}.log')
        logger = setup_logger(f'train_{name}', epoch_log_path)
        logger.info(f"Training started (DYNAMIC MASKING, shared) | model={name} | "
                    f"epochs={args.epochs} | batch={args.batch_size} | "
                    f"lr={args.learning_rate} | weight_decay={args.weight_decay} | "
                    f"warmup_steps={warmup_steps} | total_steps={total_steps} | "
                    f"early_stopping_patience={args.early_stopping_patience} | "
                    f"hidden={args.hidden_size} | layers={args.num_layers}")

        tb_dir       = os.path.join(log_dir, 'tb')
        train_writer = tf.summary.create_file_writer(os.path.join(tb_dir, 'train'))
        val_writer   = tf.summary.create_file_writer(os.path.join(tb_dir, 'val'))

        model_states[name] = {
            'model':         model,
            'color':         color,
            'optimizer':     optimizer,
            'lr_schedule':   lr_schedule,
            'early_stop':    EarlyStopping(patience=args.early_stopping_patience,
                                           min_delta=1e-4),
            'logger':        logger,
            'log_dir':       log_dir,
            'epoch_csv_path':epoch_csv_path,
            'step_log_path': step_log_path,
            'train_writer':  train_writer,
            'val_writer':    val_writer,
            'global_step':   0,
            'best_val_loss': float('inf'),
            'best_epoch':    0,
            'stopped':       False,
            'results':       {},
        }

    # ── Eager warm-up for all models ──────────────────────────────────────────
    print(f"\n{C.BOLD}Warming up models (building weights)...{C.RESET}")
    warmup_masked, warmup_labels = apply_epoch_masking(
        train_data['input_ids'][:args.batch_size * 2],
        train_data['attention_mask'][:args.batch_size * 2],
        seed=99999)
    warmup_ds = make_masked_dataset(
        warmup_masked, train_data['attention_mask'][:args.batch_size * 2],
        warmup_labels, args.batch_size, shuffle=False)
    _wb = next(iter(warmup_ds))

    for name, ms in model_states.items():
        model     = ms['model']
        optimizer = ms['optimizer']
        with tf.GradientTape() as _t:
            _lo = model(_wb, training=True)
            _ls, _, _ptl = mlm_loss_acc(_lo, _wb['labels'])
        _g = _t.gradient(_ls, model.trainable_variables)
        optimizer.apply_gradients(zip(_g, model.trainable_variables))
        n_params = sum(int(tf.size(v).numpy()) for v in model.trainable_variables)
        optimizer.iterations.assign(0)
        print(f"  {ms['color']}[{name}]{C.RESET} params: {n_params:,}")
    del _wb, warmup_ds, warmup_masked, warmup_labels

    # ══════════════════════════════════════════════════════════════════════════
    #  EPOCH-BASED TRAINING LOOP
    #  Each epoch: mask once → train ALL models on same data → validate all
    # ══════════════════════════════════════════════════════════════════════════

    global_max_step_reached = False

    for epoch in range(1, args.epochs + 1):
        if global_max_step_reached:
            break

        # Check if all models have stopped
        active = [n for n, ms in model_states.items() if not ms['stopped']]
        if not active:
            print(f"\n  {C.YELLOW}All models early-stopped.{C.RESET}")
            break

        # ── 1. Apply fresh masks for this epoch (SHARED across all models) ───
        epoch_seed = epoch * 1000 + 42  # deterministic, different each epoch
        masked_ids, labels = apply_epoch_masking(
            train_data['input_ids'], train_data['attention_mask'],
            seed=epoch_seed)
        train_ds = make_masked_dataset(
            masked_ids, train_data['attention_mask'], labels,
            args.batch_size, shuffle=True, seed=epoch)

        n_masked = (labels != -100).sum()
        n_total  = (train_data['attention_mask'] == 1).sum()
        actual_mask_pct = n_masked / max(n_total, 1) * 100

        print(f"\n{C.BOLD}--- Epoch {epoch} --- "
              f"(mask seed={epoch_seed}, "
              f"masked={n_masked:,}/{n_total:,} = {actual_mask_pct:.1f}%)"
              f"{C.RESET}")

        # ── 2. Train each active model on the SAME masked dataset ────────────
        # We iterate the dataset once, collecting all batches, then feed to
        # each model. This ensures identical batch ordering.
        batches = list(train_ds)

        for name in active:
            ms = model_states[name]
            model     = ms['model']
            optimizer = ms['optimizer']
            color     = ms['color']

            tr_losses, tr_accs = [], []
            t0 = time.time()

            for batch in batches:
                loss, acc = train_step(model, optimizer, batch)
                tr_losses.append(float(loss))
                tr_accs.append(float(acc))
                ms['global_step'] += 1
                gs = ms['global_step']

                if args.log_every > 0 and gs % args.log_every == 0:
                    with ms['train_writer'].as_default():
                        tf.summary.scalar('lm_example_loss',    loss, step=gs)
                        tf.summary.scalar('masked_lm_accuracy', acc,  step=gs)
                        tf.summary.scalar('perplexity',
                                          tf.exp(tf.minimum(loss, 20.0)), step=gs)
                    save_step_log(ms['step_log_path'], name, gs, loss, acc)

                if (args.log_every > 0 and gs % (args.log_every * 5) == 0):
                    print(f"    {color}[{name}]{C.RESET}"
                          f"  step={gs:6d}"
                          f"  loss={float(loss):.4f}"
                          f"  acc={float(acc):.4f}"
                          f"  ppl={math.exp(min(float(loss), 20)):.2f}")

                if args.max_steps > 0 and gs >= args.max_steps:
                    global_max_step_reached = True
                    break

            elapsed = time.time() - t0
            n_steps = max(len(tr_losses), 1)
            sps     = n_steps / max(elapsed, 1e-6)

            # ── 3. Validation ─────────────────────────────────────────────────
            va_losses, va_accs = [], []
            for batch in val_ds:
                vl, va = eval_step(model, batch)
                va_losses.append(float(vl))
                va_accs.append(float(va))

            t_loss = sum(tr_losses) / n_steps
            t_acc  = sum(tr_accs)  / n_steps
            v_loss = sum(va_losses) / max(len(va_losses), 1)
            v_acc  = sum(va_accs)  / max(len(va_accs),  1)
            v_ppl  = math.exp(min(v_loss, 20))
            t_ppl  = math.exp(min(t_loss, 20))

            with ms['val_writer'].as_default():
                tf.summary.scalar('lm_example_loss',    v_loss, step=ms['global_step'])
                tf.summary.scalar('masked_lm_accuracy', v_acc,  step=ms['global_step'])
                tf.summary.scalar('perplexity',         v_ppl,  step=ms['global_step'])
                tf.summary.scalar('steps_per_second',   sps,    step=ms['global_step'])

            current_lr = float(ms['lr_schedule'](ms['optimizer'].iterations))
            log_epoch(ms['logger'], name, epoch, args.epochs,
                      t_loss, t_acc, v_loss, v_acc, sps, lr=current_lr)

            append_epoch_csv(ms['epoch_csv_path'], {
                'epoch':           epoch,
                'global_step':     ms['global_step'],
                'learning_rate':   round(current_lr, 8),
                'train_loss':      round(t_loss, 6),
                'train_acc':       round(t_acc,  6),
                'train_ppl':       round(t_ppl,  4),
                'val_loss':        round(v_loss,  6),
                'val_acc':         round(v_acc,   6),
                'val_ppl':         round(v_ppl,   4),
                'steps_per_second':round(sps,     4),
                'elapsed_s':       round(elapsed, 2),
            })

            print(f"  {color}[{name}]{C.RESET}"
                  f"  Epoch {epoch:3d}"
                  f"  | LR: {current_lr:.3e}"
                  f"  | Train: loss={t_loss:.4f}  acc={t_acc:.4f}  ppl={t_ppl:.2f}"
                  f"  | Val:   loss={v_loss:.4f}  acc={v_acc:.4f}  ppl={v_ppl:.2f}"
                  f"  | {sps:.1f} step/s")

            if v_loss < ms['best_val_loss']:
                ms['best_val_loss'] = v_loss
                ms['best_epoch']    = epoch
                model.save_weights(os.path.join(ms['log_dir'],
                                                'best_model.weights.h5'))
                ms['logger'].info(f"Saved best model at epoch {epoch} "
                                  f"with val_loss={v_loss:.4f}")
                print(f"    {C.GREEN}New best val_loss={v_loss:.4f}  "
                      f"(epoch {epoch}){C.RESET}")

            ms['results'] = dict(
                epoch=epoch,          global_step=ms['global_step'],
                train_loss=t_loss,    train_acc=t_acc,    train_ppl=t_ppl,
                val_loss=v_loss,      val_acc=v_acc,      val_ppl=v_ppl,
                best_val_loss=ms['best_val_loss'], best_epoch=ms['best_epoch'],
                steps_per_second=sps)

            # Early stopping check
            if args.early_stopping_patience > 0 and ms['early_stop'].step(v_loss):
                print(f"    {C.YELLOW}[{name}] Early stopping triggered "
                      f"(patience={args.early_stopping_patience}){C.RESET}")
                ms['logger'].info(f"Early stopping at epoch {epoch}")
                ms['stopped'] = True

        # Free epoch data
        del masked_ids, labels, train_ds, batches

        if global_max_step_reached:
            print(f"\n  {C.YELLOW}max_steps={args.max_steps} reached.{C.RESET}")
            break

    # ══════════════════════════════════════════════════════════════════════════
    #  POST-TRAINING: Test evaluation + latency + comparison
    # ══════════════════════════════════════════════════════════════════════════

    results = {}
    for name, ms in model_states.items():
        model = ms['model']
        color = ms['color']

        # Load best checkpoint
        best_ckpt = os.path.join(ms['log_dir'], 'best_model.weights.h5')
        if os.path.exists(best_ckpt):
            model.load_weights(best_ckpt)
            print(f"  {color}Evaluating {name} on test set "
                  f"(best epoch {ms['best_epoch']})...{C.RESET}")

        te_losses, te_accs = [], []
        for batch in test_ds:
            tl, ta = eval_step(model, batch)
            te_losses.append(float(tl))
            te_accs.append(float(ta))
        te_loss = sum(te_losses) / max(len(te_losses), 1)
        te_acc  = sum(te_accs)   / max(len(te_accs), 1)
        te_ppl  = math.exp(min(te_loss, 20))

        ms['results']['test_loss'] = te_loss
        ms['results']['test_acc']  = te_acc
        ms['results']['test_ppl']  = te_ppl
        print(f"  {color}[{name}] Test: loss={te_loss:.4f}  "
              f"acc={te_acc:.4f}  ppl={te_ppl:.2f}{C.RESET}")

        # Forward-pass latency
        print(f"  {color}Measuring {name} forward-pass latency...{C.RESET}")
        lat_masked, lat_labels = apply_epoch_masking(
            train_data['input_ids'][:args.batch_size * 2],
            train_data['attention_mask'][:args.batch_size * 2], seed=77777)
        lat_ds = make_masked_dataset(
            lat_masked, train_data['attention_mask'][:args.batch_size * 2],
            lat_labels, args.batch_size, shuffle=False)
        sample_batch = next(iter(lat_ds))
        for _ in range(3):
            _out = model(sample_batch, training=False)
            tf.reduce_sum(_out).numpy()
        _lat_times = []
        for _ in range(10):
            _t0 = time.perf_counter()
            _out = model(sample_batch, training=False)
            tf.reduce_sum(_out).numpy()
            _lat_times.append((time.perf_counter() - _t0) * 1000)
        latency_ms = sum(_lat_times) / len(_lat_times)
        ms['results']['forward_latency_ms'] = round(latency_ms, 3)
        print(f"  {color}[{name}] Latency: {latency_ms:.2f} ms / batch{C.RESET}")

        results[name] = ms['results']
        run_logger.info(
            f"[{name}] finished | val_loss={ms['results']['val_loss']:.4f} | "
            f"test_loss={te_loss:.4f} | best_epoch={ms['best_epoch']}")

    print_comparison(results,
                     threshold=args.go_threshold,
                     speed_threshold=args.speed_threshold)
    save_results_json(results, args.output_dir, args)
    save_results_csv(results, args.output_dir)

    run_logger.info("Run complete (dynamic masking, shared masks).")

    print(f"\n  {C.CYAN}TensorBoard:{C.RESET}")
    print(f"    tensorboard --logdir {args.output_dir} --port 6006")
    print(f"\n  {C.CYAN}Log files:{C.RESET}")
    for name in results:
        log_dir = os.path.join(args.output_dir, name)
        print(f"    {name:12s}: {log_dir}/training-{name}.log")


if __name__ == "__main__":
    main()
