#!/usr/bin/env python3
# =============================================================================
# ablation_drop_budgets.py  —  Drop Budget Ablation Study
#
# Tests alternative ProgDrop ratio configurations at seq_len=512 to find
# the optimal trade-off between quality (val loss) and efficiency (FLOPs).
#
# Budget configurations tested:
#   - Aggressive : 90/60/30  (k1=461, k2=307, k3=154)
#   - Default    : 75/50/25  (k1=384, k2=256, k3=128)  ← baseline
#   - Conservative: 85/65/40 (k1=435, k2=333, k3=205)
#
# Each config trains a ProgDrop BERT-base model plus a shared Vanilla baseline.
# Results are saved as JSON/CSV and a comparison chart is generated.
#
# Usage:
#   python scripts/ablation_drop_budgets.py \
#     --data_path /data/umityilmaz/token_drop_v2/data/wikitext_mlm_512.csv \
#     --output_dir /data/umityilmaz/token_drop_v2/checkpoints/ablation_budgets
#
# Quick smoke test (200 steps):
#   python scripts/ablation_drop_budgets.py --max_steps 200 --batch_size 4
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

# ── Repo path ────────────────────────────────────────────────────────────────
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'experiments', 'progressive_contextual_dropping'))
sys.path.insert(0, REPO)   # highest priority: repo root

import numpy as np
import tensorflow as tf

# Reuse FLOPs computation from existing script
sys.path.insert(0, os.path.join(REPO, 'scripts'))
from compute_flops import (
    compute_vanilla_flops,
    compute_progressive_flops,
    format_flops,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── ANSI colours ─────────────────────────────────────────────────────────────
class C:
    BOLD    = '\033[1m'
    RESET   = '\033[0m'
    BLUE    = '\033[94m'
    YELLOW  = '\033[93m'
    GREEN   = '\033[92m'
    RED     = '\033[91m'
    CYAN    = '\033[96m'
    MAGENTA = '\033[95m'

VOCAB_SIZE = 30522   # bert-base-uncased


# ── Budget configurations ────────────────────────────────────────────────────

BUDGET_CONFIGS = {
    'aggressive': {
        'label': 'Aggressive (90/60/30)',
        'ratios': (0.90, 0.60, 0.30),
        'color': C.RED,
        'plot_color': '#E74C3C',
    },
    'default': {
        'label': 'Default (75/50/25)',
        'ratios': (0.75, 0.50, 0.25),
        'color': C.GREEN,
        'plot_color': '#27AE60',
    },
    'conservative': {
        'label': 'Conservative (85/65/40)',
        'ratios': (0.85, 0.65, 0.40),
        'color': C.YELLOW,
        'plot_color': '#F39C12',
    },
}


def ratios_to_k(ratios, seq_len):
    """Convert keep-ratio tuple (r1, r2, r3) to integer token counts."""
    k1 = int(round(ratios[0] * seq_len))
    k2 = int(round(ratios[1] * seq_len))
    k3 = int(round(ratios[2] * seq_len))
    # Ensure strict ordering k3 < k2 < k1 < seq_len
    k1 = min(k1, seq_len - 1)
    k2 = min(k2, k1 - 1)
    k3 = min(k3, k2 - 1)
    return k1, k2, k3


# ── LR Schedule ──────────────────────────────────────────────────────────────

class WarmupLinearDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup then linear decay to 0."""

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


# ── Early Stopping ───────────────────────────────────────────────────────────

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


# ── Data loading ─────────────────────────────────────────────────────────────

def load_csv_dataset(path, max_samples=0):
    """Load MLM CSV → dict of numpy arrays."""
    print(f"  Loading data from {path} ...", flush=True)
    input_ids, attention_mask, labels = [], [], []

    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_samples > 0 and i >= max_samples:
                break
            input_ids.append(json.loads(row['input_ids']))
            attention_mask.append(json.loads(row['attention_mask']))
            labels.append(json.loads(row['labels']))

    data = {
        'input_ids':      np.array(input_ids, dtype=np.int32),
        'attention_mask':  np.array(attention_mask, dtype=np.int32),
        'labels':         np.array(labels, dtype=np.int32),
    }
    print(f"  Loaded {len(input_ids):,} samples, seq_len={data['input_ids'].shape[1]}")
    return data


def make_tf_datasets(data, batch_size, val_ratio=0.05, test_ratio=0.05):
    """Split data → train/val/test tf.data.Datasets."""
    n = len(data['input_ids'])
    indices = np.random.permutation(n)

    n_val  = int(n * val_ratio)
    n_test = int(n * test_ratio)
    n_train = n - n_val - n_test

    splits = {}
    for name, start, end in [('train', 0, n_train),
                              ('val', n_train, n_train + n_val),
                              ('test', n_train + n_val, n)]:
        idx = indices[start:end]
        d = {k: v[idx] for k, v in data.items()}
        ds = tf.data.Dataset.from_tensor_slices(d)
        if name == 'train':
            ds = ds.shuffle(min(len(idx), 50000))
        ds = ds.batch(batch_size, drop_remainder=(name == 'train'))
        ds = ds.prefetch(tf.data.AUTOTUNE)
        splits[name] = ds

    print(f"  Splits: train={n_train:,}  val={n_val:,}  test={n_test:,}")
    return splits['train'], splits['val'], splits['test']


# ── MLM loss ─────────────────────────────────────────────────────────────────

def mlm_loss_acc(logits, labels):
    mask     = tf.cast(tf.not_equal(labels, -100), tf.float32)
    safe_lbl = tf.where(labels == -100, tf.zeros_like(labels), labels)
    per_tok  = tf.keras.losses.sparse_categorical_crossentropy(
        safe_lbl, tf.cast(logits, tf.float32), from_logits=True)
    n    = tf.reduce_sum(mask) + 1e-9
    loss = tf.reduce_sum(per_tok * mask) / n
    preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
    acc   = tf.reduce_sum(tf.cast(tf.equal(preds, safe_lbl), tf.float32) * mask) / n
    return loss, acc


# ── BertMLM wrapper ──────────────────────────────────────────────────────────

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


# ── Model builder ────────────────────────────────────────────────────────────

def build_vanilla(args):
    """Build Vanilla BERT baseline."""
    from official.nlp.modeling.networks import bert_encoder as _bert_enc_mod
    enc = _bert_enc_mod.BertEncoder(
        vocab_size          = VOCAB_SIZE,
        hidden_size         = args.hidden_size,
        num_layers          = args.num_layers,
        num_attention_heads = args.num_heads,
        inner_dim           = args.intermediate_size,
        max_sequence_length = args.max_seq_len,
        output_dropout      = args.dropout_rate,
        attention_dropout   = args.dropout_rate,
        dict_outputs        = True)
    return BertMLM(enc, VOCAB_SIZE, args.hidden_size, name='vanilla_mlm')


def build_progressive(args, k1, k2, k3, model_name='progressive_mlm'):
    """Build ProgDrop BERT with given budget."""
    from experiments.progressive_contextual_dropping.encoder import (
        ProgressiveContextualDropEncoder)
    enc = ProgressiveContextualDropEncoder(
        vocab_size          = VOCAB_SIZE,
        hidden_size         = args.hidden_size,
        num_layers          = args.num_layers,
        num_attention_heads = args.num_heads,
        inner_dim           = args.intermediate_size,
        max_sequence_length = args.max_seq_len,
        output_dropout      = args.dropout_rate,
        attention_dropout   = args.dropout_rate,
        token_keep_k1       = k1,
        token_keep_k2       = k2,
        token_keep_k3       = k3)
    return BertMLM(enc, VOCAB_SIZE, args.hidden_size, name=model_name)


# ── Training infrastructure ──────────────────────────────────────────────────

@tf.function
def train_step(model, optimizer, batch):
    with tf.GradientTape() as tape:
        logits       = model(batch, training=True)
        loss, acc    = mlm_loss_acc(logits, batch['labels'])
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, acc


@tf.function
def eval_step(model, batch):
    logits    = model(batch, training=False)
    loss, acc = mlm_loss_acc(logits, batch['labels'])
    return loss, acc


def setup_logger(name, log_file):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers.clear()
    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter('%(asctime)s  %(message)s',
                                       datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(fh)
    return logger


def evaluate(model, dataset):
    """Run evaluation, return (loss, acc)."""
    total_loss, total_acc, n_batches = 0.0, 0.0, 0
    for batch in dataset:
        loss, acc = eval_step(model, batch)
        total_loss += loss.numpy()
        total_acc  += acc.numpy()
        n_batches  += 1
    return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1)


def train_model(name, model, train_ds, val_ds, args, log_dir,
                total_steps, warmup_steps, color):
    """Full training loop for a single model. Returns results dict."""
    os.makedirs(log_dir, exist_ok=True)
    print(f"\n{C.BOLD}{'═' * 65}{C.RESET}")
    print(f"  {color}{C.BOLD}▶  Training: {name}{C.RESET}")
    print(f"{C.BOLD}{'═' * 65}{C.RESET}")

    logger = setup_logger(f'ablation_{name}', os.path.join(log_dir, f'training-{name}.log'))
    logger.info(f"Training started | model={name} | epochs={args.epochs} | "
                f"batch={args.batch_size} | lr={args.learning_rate}")

    csv_path = os.path.join(log_dir, f'training-{name}.csv')
    csv_f    = open(csv_path, 'w', newline='')
    csv_w    = csv.writer(csv_f)
    csv_w.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc',
                     'epoch_time', 'steps_per_sec'])

    lr_schedule = WarmupLinearDecay(
        peak_lr=args.learning_rate, warmup_steps=warmup_steps,
        total_steps=max(total_steps, 1))
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule, weight_decay=args.weight_decay,
        beta_1=0.9, beta_2=0.999, epsilon=1e-8)

    early_stop    = EarlyStopping(patience=args.early_stopping_patience, min_delta=1e-4)
    best_val_loss = float('inf')
    best_epoch    = 0
    global_step   = 0

    # Eager warm-up
    print(f"  Warming up '{name}' (building weights)...", flush=True)
    _wb = next(iter(train_ds))
    with tf.GradientTape() as _t:
        _lo = model(_wb, training=True)
        _ls, _, = mlm_loss_acc(_lo, _wb['labels'])
    _t.gradient(_ls, model.trainable_variables)
    n_params = sum(np.prod(v.shape) for v in model.trainable_variables)
    print(f"  {name}: {n_params:,} trainable parameters")

    best_weights_path = os.path.join(log_dir, f'best_{name}.weights.h5')

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        epoch_loss, epoch_acc, steps = 0.0, 0.0, 0

        for batch in train_ds:
            loss, acc = train_step(model, optimizer, batch)
            epoch_loss += loss.numpy()
            epoch_acc  += acc.numpy()
            steps      += 1
            global_step += 1

            if args.log_every > 0 and global_step % args.log_every == 0:
                avg_l = epoch_loss / steps
                avg_a = epoch_acc / steps
                print(f"  [{name}] step {global_step:>6d}  "
                      f"loss={avg_l:.4f}  acc={avg_a:.4f}", flush=True)

            if args.max_steps > 0 and global_step >= args.max_steps:
                break

        elapsed = time.time() - t0
        train_loss = epoch_loss / max(steps, 1)
        train_acc  = epoch_acc / max(steps, 1)
        sps        = steps / max(elapsed, 0.01)

        val_loss, val_acc = evaluate(model, val_ds)
        ppl = math.exp(min(val_loss, 20))

        improved = '★' if val_loss < best_val_loss else ' '
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch
            model.save_weights(best_weights_path)

        msg = (f"Epoch {epoch:>3d}  "
               f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
               f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
               f"ppl={ppl:.2f}  time={elapsed:.1f}s  sps={sps:.2f} {improved}")
        logger.info(msg)
        print(f"  {color}{msg}{C.RESET}")

        csv_w.writerow([epoch, f'{train_loss:.6f}', f'{train_acc:.6f}',
                        f'{val_loss:.6f}', f'{val_acc:.6f}',
                        f'{elapsed:.1f}', f'{sps:.3f}'])
        csv_f.flush()

        if args.early_stopping_patience > 0 and early_stop.step(val_loss):
            print(f"  {color}Early stopping at epoch {epoch} "
                  f"(patience={args.early_stopping_patience}){C.RESET}")
            break

        if args.max_steps > 0 and global_step >= args.max_steps:
            print(f"  {color}Reached max_steps={args.max_steps}{C.RESET}")
            break

    csv_f.close()

    # Reload best weights for test evaluation
    if os.path.exists(best_weights_path):
        model.load_weights(best_weights_path)

    return {
        'name':          name,
        'best_val_loss': best_val_loss,
        'best_epoch':    best_epoch,
        'final_step':    global_step,
        'params':        n_params,
    }


# ── FLOPs computation ────────────────────────────────────────────────────────

def compute_config_flops(args, k1, k2, k3):
    """Compute vanilla and progressive FLOPs for given budget."""
    van_f, _ = compute_vanilla_flops(
        args.num_layers, args.max_seq_len,
        args.hidden_size, args.intermediate_size, args.num_heads)
    prog_f, _ = compute_progressive_flops(
        args.num_layers, args.max_seq_len, k1, k2, k3,
        args.hidden_size, args.intermediate_size, args.num_heads)
    return van_f, prog_f


# ── Result plots ─────────────────────────────────────────────────────────────

def plot_ablation_results(results, output_dir):
    """Generate comparison charts for ablation study."""

    # ── Bar chart: Val Loss vs FLOPs savings ─────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    names   = [r['label'] for r in results]
    losses  = [r['best_val_loss'] for r in results]
    savings = [r['flop_savings_pct'] for r in results]
    colors  = [r['plot_color'] for r in results]

    # Val loss comparison
    bars1 = ax1.bar(names, losses, color=colors, edgecolor='white', width=0.55)
    ax1.set_ylabel('Best Validation Loss', fontsize=12)
    ax1.set_title('Quality: Validation Loss by Budget', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    for bar, val in zip(bars1, losses):
        ax1.annotate(f'{val:.4f}',
                     xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     xytext=(0, 5), textcoords='offset points',
                     ha='center', va='bottom', fontsize=11, fontweight='bold')

    # FLOP savings comparison
    bars2 = ax2.bar(names, savings, color=colors, edgecolor='white', width=0.55)
    ax2.set_ylabel('FLOP Savings vs Vanilla (%)', fontsize=12)
    ax2.set_title('Efficiency: FLOP Reduction by Budget', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    for bar, val in zip(bars2, savings):
        ax2.annotate(f'{val:.1f}%',
                     xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     xytext=(0, 5), textcoords='offset points',
                     ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    chart_path = os.path.join(output_dir, 'ablation_budget_comparison.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Chart saved: {chart_path}")

    # ── Scatter: quality vs efficiency trade-off ─────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in results:
        ax.scatter(r['flop_savings_pct'], r['best_val_loss'],
                   color=r['plot_color'], s=200, zorder=5, edgecolors='black')
        ax.annotate(r['label'],
                    xy=(r['flop_savings_pct'], r['best_val_loss']),
                    xytext=(10, 5), textcoords='offset points',
                    fontsize=10, fontweight='bold')

    # Add vanilla reference line
    if results:
        van_loss = results[0].get('vanilla_val_loss', None)
        if van_loss:
            ax.axhline(y=van_loss, color='#5B9BD5', linestyle='--',
                       alpha=0.7, label=f'Vanilla baseline ({van_loss:.4f})')
            ax.legend(fontsize=10)

    ax.set_xlabel('FLOP Savings vs Vanilla (%)', fontsize=12)
    ax.set_ylabel('Best Validation Loss', fontsize=12)
    ax.set_title('Quality vs Efficiency Trade-off', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    scatter_path = os.path.join(output_dir, 'ablation_budget_tradeoff.png')
    plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Chart saved: {scatter_path}")

    # ── Training curves comparison ───────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for r in results:
        csv_path = r.get('csv_path')
        if csv_path and os.path.exists(csv_path):
            epochs, train_losses, val_losses = [], [], []
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    epochs.append(int(row['epoch']))
                    train_losses.append(float(row['train_loss']))
                    val_losses.append(float(row['val_loss']))
            if epochs:
                ax1.plot(epochs, train_losses, color=r['plot_color'],
                         label=r['label'], linewidth=2)
                ax2.plot(epochs, val_losses, color=r['plot_color'],
                         label=r['label'], linewidth=2)

    # Plot vanilla baseline curve if available
    van_csv = os.path.join(output_dir, 'vanilla', 'training-vanilla.csv')
    if os.path.exists(van_csv):
        epochs, tl, vl = [], [], []
        with open(van_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append(int(row['epoch']))
                tl.append(float(row['train_loss']))
                vl.append(float(row['val_loss']))
        if epochs:
            ax1.plot(epochs, tl, color='#5B9BD5', label='Vanilla',
                     linewidth=2, linestyle='--')
            ax2.plot(epochs, vl, color='#5B9BD5', label='Vanilla',
                     linewidth=2, linestyle='--')

    for ax, title in [(ax1, 'Training Loss'), (ax2, 'Validation Loss')]:
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    curves_path = os.path.join(output_dir, 'ablation_budget_curves.png')
    plt.savefig(curves_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Chart saved: {curves_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Ablation study on ProgDrop budget configurations (seq=512)')

    # Data
    p.add_argument('--data_path', default=
        '/data/umityilmaz/token_drop_v2/data/wikitext_mlm_512.csv')
    p.add_argument('--output_dir', default=
        '/data/umityilmaz/token_drop_v2/checkpoints/ablation_budgets')
    p.add_argument('--max_samples', type=int, default=0,
                   help='Limit dataset rows (0 = all)')
    p.add_argument('--val_ratio',   type=float, default=0.05)
    p.add_argument('--test_ratio',  type=float, default=0.05)

    # Training
    p.add_argument('--epochs',        type=int,   default=50000)
    p.add_argument('--max_steps',     type=int,   default=200000,
                   help='Stop after N steps (0 = epoch-based)')
    p.add_argument('--batch_size',    type=int,   default=16)
    p.add_argument('--learning_rate', type=float, default=1e-4)
    p.add_argument('--log_every',     type=int,   default=500)

    # Regularization
    p.add_argument('--weight_decay',            type=float, default=0.01)
    p.add_argument('--warmup_ratio',            type=float, default=0.06)
    p.add_argument('--early_stopping_patience', type=int,   default=10)

    # Architecture (BERT-base defaults for seq=512)
    p.add_argument('--hidden_size',       type=int,   default=768)
    p.add_argument('--num_layers',        type=int,   default=12)
    p.add_argument('--num_heads',         type=int,   default=12)
    p.add_argument('--intermediate_size', type=int,   default=3072)
    p.add_argument('--max_seq_len',       type=int,   default=512)
    p.add_argument('--dropout_rate',      type=float, default=0.1)

    # Budget configs to test
    p.add_argument('--configs', nargs='+',
                   default=['aggressive', 'default', 'conservative'],
                   choices=list(BUDGET_CONFIGS.keys()),
                   help='Which budget configs to test')
    p.add_argument('--skip_vanilla', action='store_true',
                   help='Skip vanilla baseline training (use if already have results)')

    return p.parse_args()


def main():
    args = parse_args()

    # GPU memory growth
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Banner ───────────────────────────────────────────────────────────────
    print(f"\n{C.CYAN}{C.BOLD}{'─'*65}")
    print(f"  Drop Budget Ablation Study")
    print(f"  seq_len={args.max_seq_len}  layers={args.num_layers}  "
          f"hidden={args.hidden_size}")
    print(f"{'─'*65}{C.RESET}")

    # Print budget configurations
    print(f"\n  Budget configurations to test:")
    for cfg_name in args.configs:
        cfg = BUDGET_CONFIGS[cfg_name]
        k1, k2, k3 = ratios_to_k(cfg['ratios'], args.max_seq_len)
        print(f"    {cfg['color']}{cfg['label']:30s}  "
              f"k1={k1}  k2={k2}  k3={k3}{C.RESET}")

    # ── Load data ────────────────────────────────────────────────────────────
    data = load_csv_dataset(args.data_path, args.max_samples)
    actual_seq_len = data['input_ids'].shape[1]
    if actual_seq_len != args.max_seq_len:
        print(f"  {C.YELLOW}Warning: data seq_len={actual_seq_len} != "
              f"max_seq_len={args.max_seq_len}. Using data seq_len.{C.RESET}")
        args.max_seq_len = actual_seq_len

    np.random.seed(42)
    train_ds, val_ds, test_ds = make_tf_datasets(
        data, args.batch_size, args.val_ratio, args.test_ratio)

    n_train = int(len(data['input_ids']) * (1 - args.val_ratio - args.test_ratio))
    steps_per_epoch = n_train // args.batch_size
    if args.max_steps > 0:
        total_steps = args.max_steps
    else:
        total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    print(f"\n  steps_per_epoch={steps_per_epoch}  total_steps={total_steps}  "
          f"warmup_steps={warmup_steps}")

    # ── Train vanilla baseline ───────────────────────────────────────────────
    vanilla_result = None
    if not args.skip_vanilla:
        van_dir = os.path.join(args.output_dir, 'vanilla')
        van_model = build_vanilla(args)
        van_res = train_model('vanilla', van_model, train_ds, val_ds, args,
                              van_dir, total_steps, warmup_steps, C.BLUE)

        # Evaluate on test set
        test_loss, test_acc = evaluate(van_model, test_ds)
        van_res['test_loss'] = test_loss
        van_res['test_acc']  = test_acc
        vanilla_result = van_res
        print(f"\n  {C.BLUE}Vanilla test: loss={test_loss:.4f}  acc={test_acc:.4f}{C.RESET}")

        del van_model
        tf.keras.backend.clear_session()

    # ── Train each budget configuration ──────────────────────────────────────
    all_results = []

    for cfg_name in args.configs:
        cfg = BUDGET_CONFIGS[cfg_name]
        k1, k2, k3 = ratios_to_k(cfg['ratios'], args.max_seq_len)

        model_label = f"prog_{cfg_name}"
        cfg_dir = os.path.join(args.output_dir, model_label)

        print(f"\n{C.BOLD}{'─'*65}{C.RESET}")
        print(f"  Config: {cfg['label']}  →  k1={k1}  k2={k2}  k3={k3}")
        print(f"{C.BOLD}{'─'*65}{C.RESET}")

        # Build model
        model = build_progressive(args, k1, k2, k3, model_name=f'{model_label}_mlm')

        # Train
        res = train_model(model_label, model, train_ds, val_ds, args,
                          cfg_dir, total_steps, warmup_steps, cfg['color'])

        # Test evaluation
        test_loss, test_acc = evaluate(model, test_ds)
        res['test_loss'] = test_loss
        res['test_acc']  = test_acc
        print(f"\n  {cfg['color']}{cfg['label']} test: "
              f"loss={test_loss:.4f}  acc={test_acc:.4f}{C.RESET}")

        # Compute FLOPs
        van_f, prog_f = compute_config_flops(args, k1, k2, k3)
        savings_pct = (1 - prog_f / van_f) * 100

        res.update({
            'config':           cfg_name,
            'label':            cfg['label'],
            'plot_color':       cfg['plot_color'],
            'ratios':           cfg['ratios'],
            'k1': k1, 'k2': k2, 'k3': k3,
            'vanilla_flops':    van_f,
            'prog_flops':       prog_f,
            'flop_savings_pct': savings_pct,
            'vanilla_val_loss': vanilla_result['best_val_loss'] if vanilla_result else None,
            'csv_path':         os.path.join(cfg_dir, f'training-{model_label}.csv'),
        })
        all_results.append(res)

        print(f"  FLOPs: {format_flops(prog_f)} "
              f"({savings_pct:.1f}% savings vs vanilla {format_flops(van_f)})")

        del model
        tf.keras.backend.clear_session()

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{C.CYAN}{C.BOLD}{'═'*65}")
    print(f"  ABLATION RESULTS SUMMARY")
    print(f"{'═'*65}{C.RESET}\n")

    if vanilla_result:
        print(f"  {C.BLUE}Vanilla baseline:  val_loss={vanilla_result['best_val_loss']:.4f}  "
              f"test_loss={vanilla_result.get('test_loss', 0):.4f}{C.RESET}\n")

    header = (f"  {'Config':<30s} {'Ratios':>12s} {'Val Loss':>10s} "
              f"{'Test Loss':>10s} {'FLOP Save':>10s} {'Best Ep':>8s}")
    print(header)
    print(f"  {'─'*30} {'─'*12} {'─'*10} {'─'*10} {'─'*10} {'─'*8}")

    for r in all_results:
        ratio_str = f"{r['ratios'][0]:.0%}/{r['ratios'][1]:.0%}/{r['ratios'][2]:.0%}"
        print(f"  {r['label']:<30s} {ratio_str:>12s} "
              f"{r['best_val_loss']:>10.4f} {r.get('test_loss', 0):>10.4f} "
              f"{r['flop_savings_pct']:>9.1f}% {r['best_epoch']:>8d}")

    # Identify best config
    best = min(all_results, key=lambda r: r['best_val_loss'])
    most_efficient = max(all_results, key=lambda r: r['flop_savings_pct'])

    print(f"\n  Best quality:    {best['label']} (val_loss={best['best_val_loss']:.4f})")
    print(f"  Most efficient:  {most_efficient['label']} "
          f"({most_efficient['flop_savings_pct']:.1f}% savings)")

    # Quality-efficiency score (lower is better): normalized loss + (1 - savings/100)
    if vanilla_result:
        for r in all_results:
            delta = r['best_val_loss'] - vanilla_result['best_val_loss']
            r['quality_score'] = delta + (1 - r['flop_savings_pct'] / 100)
        best_tradeoff = min(all_results, key=lambda r: r['quality_score'])
        print(f"  Best trade-off:  {best_tradeoff['label']}")

    # ── Save results ─────────────────────────────────────────────────────────
    summary = {
        'timestamp': datetime.datetime.now().isoformat(),
        'args': {k: v for k, v in vars(args).items() if not k.startswith('_')},
        'vanilla': vanilla_result,
        'configs': [],
    }
    for r in all_results:
        entry = {k: v for k, v in r.items()
                 if k not in ('plot_color', 'csv_path')}
        # Convert numpy types to native python
        for k, v in entry.items():
            if isinstance(v, (np.integer,)):
                entry[k] = int(v)
            elif isinstance(v, (np.floating,)):
                entry[k] = float(v)
        summary['configs'].append(entry)

    json_path = os.path.join(args.output_dir, 'ablation_results.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved: {json_path}")

    # CSV summary
    csv_path = os.path.join(args.output_dir, 'ablation_results.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['config', 'label', 'ratios', 'k1', 'k2', 'k3',
                     'best_val_loss', 'test_loss', 'flop_savings_pct', 'best_epoch'])
        for r in all_results:
            ratio_str = f"{r['ratios'][0]}/{r['ratios'][1]}/{r['ratios'][2]}"
            w.writerow([r['config'], r['label'], ratio_str,
                        r['k1'], r['k2'], r['k3'],
                        f"{r['best_val_loss']:.6f}",
                        f"{r.get('test_loss', 0):.6f}",
                        f"{r['flop_savings_pct']:.2f}",
                        r['best_epoch']])
    print(f"  Results saved: {csv_path}")

    # ── Generate plots ───────────────────────────────────────────────────────
    plot_ablation_results(all_results, args.output_dir)

    print(f"\n{C.CYAN}{C.BOLD}{'═'*65}")
    print(f"  Ablation study complete!")
    print(f"{'═'*65}{C.RESET}\n")


if __name__ == '__main__':
    main()
