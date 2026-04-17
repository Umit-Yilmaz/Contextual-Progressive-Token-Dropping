#!/usr/bin/env python3
# =============================================================================
# train_csv_comparison.py  —  3-Way BERT Comparison (CSV edition)
#
# Trains Vanilla BERT, TokenDrop BERT, and Progressive Drop BERT
# using the existing bookcorpus_static_mlm.csv dataset from prior experiments.
# No TFRecord pipeline required — works directly with the CSV data.
#
# Architecture: BERT-Mini (hidden=256, layers=4, heads=4) matching experiment_2
#
# Usage:
#   python scripts/train_csv_comparison.py \
#     --data_path /data/umityilmaz/experiment_1/vanilla-bert/bookcorpus_static_mlm.csv \
#     --output_dir /data/umityilmaz/token_drop_l2/checkpoints/train_csv \
#     --epochs 10 --batch_size 64
#
# Quick smoke test (10 steps):
#   python scripts/train_csv_comparison.py --max_steps 10 --batch_size 4
#
# Single model:
#   python scripts/train_csv_comparison.py --models progressive --epochs 5
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
# REPO must come before the subpackage so `import encoder` finds the ROOT encoder
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

VOCAB_SIZE = 30522   # bert-base-uncased vocab


# ── LR Schedule: linear warmup → linear decay to 0 ───────────────────────────

class WarmupLinearDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup for `warmup_steps`, then linear decay to 0 at `total_steps`."""

    def __init__(self, peak_lr: float, warmup_steps: int, total_steps: int):
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
        return {'peak_lr':      self.peak_lr,
                'warmup_steps': self.warmup_steps,
                'total_steps':  self.total_steps}


# ── Early Stopping ────────────────────────────────────────────────────────────

class EarlyStopping:
    """Stop training when val_loss does not improve for `patience` epochs."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best      = float('inf')

    def step(self, val_loss: float) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best - self.min_delta:
            self.best    = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

    def reset(self):
        self.counter = 0
        self.best    = float('inf')


# ── Logging helpers ────────────────────────────────────────────────────────────

def setup_logger(name, log_file):
    """Create a logger that writes to both a file and stdout."""
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

    # Stream handler without timestamps (colours shown by print in the loop)
    return logger


def log_epoch(logger, name, epoch, n_epochs, t_loss, t_acc, v_loss, v_acc, sps,
              lr=None):
    """Write an epoch summary line matching experiment_2 log format."""
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
    """Append one epoch row to the per-model epoch CSV."""
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def save_step_log(log_path, model_name, global_step, loss, acc):
    """Append a step-level log line."""
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"{ts} | INFO | [{model_name}] step={global_step} "
                f"loss={float(loss):.4f} acc={float(acc):.4f} "
                f"ppl={math.exp(min(float(loss), 20)):.2f}\n")


# ── Data loading ───────────────────────────────────────────────────────────────

def load_csv_data(csv_path, max_samples=None, val_ratio=0.05, test_ratio=0.05,
                   seed=42):
    """Load bookcorpus_static_mlm.csv into numpy arrays.

    CSV format (created by HuggingFace static-MLM pipeline):
      input_ids       – token IDs; [MASK]=103 at masked positions
      attention_mask  – 1=real token, 0=padding
      labels          – -100=non-masked, true_token_id=masked

    Returns:
      (train_data, val_data, test_data) each a dict with keys
      'input_ids', 'attention_mask', 'labels'  (np.int32 arrays).
    """
    print(f"\n{C.CYAN}Loading CSV data:{C.RESET} {csv_path}")
    ids_list, mask_list, lbl_list = [], [], []

    with open(csv_path, 'r', encoding='utf-8') as fh:
        reader = csv.reader(fh)
        next(reader)   # skip header row
        for i, row in enumerate(reader):
            if max_samples and i >= max_samples:
                break
            if len(row) < 3:
                continue
            ids_list.append(json.loads(row[0]))
            mask_list.append(json.loads(row[1]))
            lbl_list.append(json.loads(row[2]))

    input_ids      = np.array(ids_list,  dtype=np.int32)
    attention_mask = np.array(mask_list, dtype=np.int32)
    labels         = np.array(lbl_list,  dtype=np.int32)
    n, seq_len     = input_ids.shape

    # Shuffle + 3-way split (train / val / test)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test  = max(1, int(n * test_ratio))
    n_val   = max(1, int(n * val_ratio))
    n_train = n - n_val - n_test

    idx_train = perm[:n_train]
    idx_val   = perm[n_train:n_train + n_val]
    idx_test  = perm[n_train + n_val:]

    train_data = dict(input_ids=input_ids[idx_train],
                      attention_mask=attention_mask[idx_train],
                      labels=labels[idx_train])
    val_data   = dict(input_ids=input_ids[idx_val],
                      attention_mask=attention_mask[idx_val],
                      labels=labels[idx_val])
    test_data  = dict(input_ids=input_ids[idx_test],
                      attention_mask=attention_mask[idx_test],
                      labels=labels[idx_test])

    masked_per_sample = (labels != -100).sum(axis=1).mean()
    print(f"  Samples  : {n:,}  (train={n_train:,}, val={n_val:,}, test={n_test:,})")
    print(f"  Seq len  : {seq_len}")
    print(f"  Masked/s : ~{masked_per_sample:.1f} tokens ({masked_per_sample/seq_len*100:.1f}%)")
    return train_data, val_data, test_data


def make_tf_dataset(data, batch_size, shuffle=True, seed=0):
    """Create a prefetched tf.data.Dataset from numpy dict."""
    ds = tf.data.Dataset.from_tensor_slices(data)
    if shuffle:
        ds = ds.shuffle(
            buffer_size=min(10_000, len(data['input_ids'])), seed=seed)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ── Loss & accuracy ────────────────────────────────────────────────────────────

def mlm_loss_acc(logits, labels):
    """Compute masked-LM loss and accuracy (ignoring label=-100 positions).

    Args:
      logits : [B, S, vocab_size]  float32
      labels : [B, S]              int32  (-100 = non-masked)

    Returns:
      loss         : scalar float32
      accuracy     : scalar float32
      per_tok_loss : [B, S] float32  (0 at non-masked positions)
    """
    mask      = tf.cast(tf.not_equal(labels, -100), tf.float32)
    safe_lbl  = tf.where(labels == -100, tf.zeros_like(labels), labels)

    per_tok_loss = tf.keras.losses.sparse_categorical_crossentropy(
        safe_lbl, tf.cast(logits, tf.float32), from_logits=True)
    n    = tf.reduce_sum(mask) + 1e-9
    loss = tf.reduce_sum(per_tok_loss * mask) / n

    preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
    acc   = tf.reduce_sum(
        tf.cast(tf.equal(preds, safe_lbl), tf.float32) * mask) / n

    return loss, acc, per_tok_loss


# ── BertMLM wrapper ────────────────────────────────────────────────────────────

class BertMLM(tf.keras.Model):
    """BERT encoder + standard MLM prediction head.

    MLM head: Dense(H→H) + approx_GeLU + LayerNorm + Dense(H→V).
    """

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
        seq_out = enc_out['sequence_output']          # [B, S, H]
        h       = self.mlm_dense(seq_out)
        h       = tf.nn.gelu(h, approximate=True)
        h       = self.mlm_norm(h)
        logits  = self.mlm_proj(h)                    # [B, S, V]
        return logits


# ── Model factory ──────────────────────────────────────────────────────────────

def build_models(args):
    """Construct all three BERT variants.

    Returns:
      list of (name, model, ansi_color)
    """
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

    # 1. Vanilla BERT ──────────────────────────────────────────────────────────
    vanilla_enc = _bert_enc_mod.BertEncoder(
        vocab_size          = VOCAB_SIZE,
        inner_dim           = args.intermediate_size,
        max_sequence_length = args.max_seq_len,
        dict_outputs        = True,
        **common)
    vanilla_model = BertMLM(vanilla_enc, VOCAB_SIZE, args.hidden_size,
                             name='vanilla_mlm')

    # 2. TokenDrop BERT ────────────────────────────────────────────────────────
    tokendrop_enc = TokenDropBertEncoder(
        vocab_size          = VOCAB_SIZE,
        intermediate_size   = args.intermediate_size,
        max_sequence_length = args.max_seq_len,
        token_keep_k        = args.token_keep_k,
        **common)
    tokendrop_model = BertMLM(tokendrop_enc, VOCAB_SIZE, args.hidden_size,
                               name='tokendrop_mlm')

    # 3. Progressive Drop BERT ─────────────────────────────────────────────────
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


# ── Training loop ──────────────────────────────────────────────────────────────

@tf.function
def train_step(model, optimizer, batch):
    with tf.GradientTape() as tape:
        logits                  = model(batch, training=True)
        loss, acc, per_tok_loss = mlm_loss_acc(logits, batch['labels'])
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Update token importance for TokenDrop encoder (vocab-level EMA).
    # Pass true token IDs at masked positions with their losses,
    # original token IDs at non-masked positions with 0 loss.
    enc = model.encoder
    if hasattr(enc, 'record_mlm_loss'):
        mlm_mask = tf.cast(tf.not_equal(batch['labels'], -100), tf.float32)
        true_ids = tf.where(
            batch['labels'] == -100, batch['input_ids'], batch['labels'])
        enc.record_mlm_loss(
            mlm_ids=true_ids,
            mlm_losses=per_tok_loss * mlm_mask)

    return loss, acc


@tf.function
def eval_step(model, batch):
    logits       = model(batch, training=False)
    loss, acc, _ = mlm_loss_acc(logits, batch['labels'])
    return loss, acc


def train_one_model(name, model, color, train_ds, val_ds, args, log_dir,
                    total_steps=0, warmup_steps=0):
    """Full training + validation loop for a single model.

    Logs to:
      {log_dir}/training-{name}.log    — epoch summaries (human-readable)
      {log_dir}/training-{name}.csv    — epoch metrics (CSV)
      {log_dir}/steps-{name}.log       — step-level loss/acc every log_every steps
      TensorBoard at {log_dir}/tb/train  and  {log_dir}/tb/val

    Returns:
      dict with final epoch metrics.
    """
    print(f"\n{C.BOLD}{'═' * 65}{C.RESET}")
    print(f"  {color}{C.BOLD}▶  Training: {name.upper()}{C.RESET}")
    print(f"{C.BOLD}{'═' * 65}{C.RESET}")

    # ── Loggers ───────────────────────────────────────────────────────────────
    epoch_log_path = os.path.join(log_dir, f'training-{name}.log')
    epoch_csv_path = os.path.join(log_dir, f'training-{name}.csv')
    step_log_path  = os.path.join(log_dir, f'steps-{name}.log')
    logger = setup_logger(f'train_{name}', epoch_log_path)
    logger.info(f"Training started | model={name} | epochs={args.epochs} | "
                f"batch={args.batch_size} | lr={args.learning_rate} | "
                f"weight_decay={args.weight_decay} | warmup_steps={warmup_steps} | "
                f"total_steps={total_steps} | "
                f"early_stopping_patience={args.early_stopping_patience} | "
                f"hidden={args.hidden_size} | layers={args.num_layers}")

    # ── TensorBoard ───────────────────────────────────────────────────────────
    tb_dir       = os.path.join(log_dir, 'tb')
    train_writer = tf.summary.create_file_writer(os.path.join(tb_dir, 'train'))
    val_writer   = tf.summary.create_file_writer(os.path.join(tb_dir, 'val'))

    # ── Optimizer: AdamW + warmup/linear-decay LR schedule ────────────────────
    lr_schedule = WarmupLinearDecay(
        peak_lr=args.learning_rate,
        warmup_steps=warmup_steps,
        total_steps=max(total_steps, 1))
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=args.weight_decay,
        beta_1=0.9, beta_2=0.999, epsilon=1e-8)

    # ── Early stopping ─────────────────────────────────────────────────────────
    early_stop    = EarlyStopping(patience=args.early_stopping_patience,
                                  min_delta=1e-4)

    global_step   = 0
    best_val_loss = float('inf')
    best_epoch    = 0
    results       = {}

    # ── Eager warm-up: force tf_keras 2.x layers to build weights before ──────
    # @tf.function tracing (Keras 3 vs tf-keras 2.x lazy-build conflict).
    print(f"  Warming up '{name}' model (building weights)...", flush=True)
    _wb = next(iter(train_ds))
    with tf.GradientTape() as _t:
        _lo = model(_wb, training=True)
        _ls, _, _ptl = mlm_loss_acc(_lo, _wb['labels'])
    _g  = _t.gradient(_ls, model.trainable_variables)
    optimizer.apply_gradients(zip(_g, model.trainable_variables))
    n_params = sum(int(tf.size(v).numpy()) for v in model.trainable_variables)
    print(f"  Warm-up done.  Trainable params: {n_params:,}", flush=True)
    # Reset iteration counter so LR schedule starts from step 0
    optimizer.iterations.assign(0)
    del _wb, _lo, _ls, _ptl, _g, _t

    for epoch in range(1, args.epochs + 1):
        # ── Training pass ─────────────────────────────────────────────────────
        tr_losses, tr_accs = [], []
        t0 = time.time()

        for batch in train_ds:
            loss, acc = train_step(model, optimizer, batch)
            tr_losses.append(float(loss))
            tr_accs.append(float(acc))
            global_step += 1

            # Step-level TensorBoard + step log
            if args.log_every > 0 and global_step % args.log_every == 0:
                with train_writer.as_default():
                    tf.summary.scalar('lm_example_loss',    loss,
                                      step=global_step)
                    tf.summary.scalar('masked_lm_accuracy', acc,
                                      step=global_step)
                    tf.summary.scalar('perplexity',
                                      tf.exp(tf.minimum(loss, 20.0)),
                                      step=global_step)
                save_step_log(step_log_path, name, global_step, loss, acc)

            # Console progress every 5×log_every steps
            if (args.log_every > 0 and
                    global_step % (args.log_every * 5) == 0):
                print(f"    {color}[{name}]{C.RESET}"
                      f"  step={global_step:6d}"
                      f"  loss={float(loss):.4f}"
                      f"  acc={float(acc):.4f}"
                      f"  ppl={math.exp(min(float(loss), 20)):.2f}")

            if args.max_steps > 0 and global_step >= args.max_steps:
                break

        elapsed = time.time() - t0
        n_steps = max(len(tr_losses), 1)
        sps     = n_steps / max(elapsed, 1e-6)

        # ── Validation pass ───────────────────────────────────────────────────
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

        # ── TensorBoard (epoch-level) ─────────────────────────────────────────
        with val_writer.as_default():
            tf.summary.scalar('lm_example_loss',    v_loss, step=global_step)
            tf.summary.scalar('masked_lm_accuracy', v_acc,  step=global_step)
            tf.summary.scalar('perplexity',         v_ppl,  step=global_step)
            tf.summary.scalar('steps_per_second',   sps,    step=global_step)

        # ── Log file (epoch summary) ──────────────────────────────────────────
        current_lr = float(lr_schedule(optimizer.iterations))
        log_epoch(logger, name, epoch, args.epochs,
                  t_loss, t_acc, v_loss, v_acc, sps, lr=current_lr)

        # ── Epoch CSV ─────────────────────────────────────────────────────────
        append_epoch_csv(epoch_csv_path, {
            'epoch':           epoch,
            'global_step':     global_step,
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

        # ── Console epoch line ─────────────────────────────────────────────────
        print(f"  {color}[{name}]{C.RESET}"
              f"  Epoch {epoch:3d}/{args.epochs}"
              f"  | LR: {current_lr:.3e}"
              f"  | Train: loss={t_loss:.4f}  acc={t_acc:.4f}  ppl={t_ppl:.2f}"
              f"  | Val:   loss={v_loss:.4f}  acc={v_acc:.4f}  ppl={v_ppl:.2f}"
              f"  | {sps:.1f} step/s")

        # ── Checkpoint best ────────────────────────────────────────────────────
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_epoch    = epoch
            model.save_weights(os.path.join(log_dir, 'best_model.weights.h5'))
            logger.info(f"Saved best model at epoch {epoch} "
                        f"with val_loss={v_loss:.4f}")
            print(f"    {C.GREEN}↓ New best val_loss={v_loss:.4f}  "
                  f"(epoch {epoch}){C.RESET}")

        results = dict(
            epoch=epoch,          global_step=global_step,
            train_loss=t_loss,    train_acc=t_acc,    train_ppl=t_ppl,
            val_loss=v_loss,      val_acc=v_acc,      val_ppl=v_ppl,
            best_val_loss=best_val_loss, best_epoch=best_epoch,
            steps_per_second=sps)

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

        # ── Early stopping check ───────────────────────────────────────────────
        if args.early_stopping_patience > 0 and early_stop.step(v_loss):
            print(f"    {C.YELLOW}[{name}] Early stopping triggered "
                  f"(patience={args.early_stopping_patience}, "
                  f"no improvement for {early_stop.patience} epochs){C.RESET}",
                  flush=True)
            logger.info(f"Early stopping at epoch {epoch} "
                        f"(best val_loss={early_stop.best:.4f})")
            break

    logger.info(f"Training complete! Best model at epoch {best_epoch} "
                f"with val_loss={best_val_loss:.4f}")
    return results


# ── Final comparison table ────────────────────────────────────────────────────

def print_comparison(results, threshold=0.10, speed_threshold=0.85):
    """Print and log the final 3-way comparison."""
    print(f"\n{C.BOLD}{'═' * 70}{C.RESET}")
    print(f"{C.BOLD}  3-Way Comparison — Final Results{C.RESET}")
    print(f"{C.BOLD}{'═' * 70}{C.RESET}")
    header = (f"  {'Model':<22} {'Val Loss':>9} {'Val Acc':>8} "
              f"{'Test Loss':>10} {'Test Acc':>9} "
              f"{'Train Loss':>10} {'Best Ep':>7} {'Steps/s':>8}")
    print(header)
    print(f"  {'─'*22} {'─'*9} {'─'*8} {'─'*10} {'─'*9} {'─'*10} {'─'*7} {'─'*8}")

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

    # Go/No-Go evaluation
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
            sym = f"{C.GREEN}✓{C.RESET}" if ok else f"{C.RED}✗{C.RESET}"
            sta = f"{C.GREEN}GO{C.RESET}" if ok else f"{C.RED}NO-GO{C.RESET}"
            print(f"  {sym}  {label:<50} [{sta}]")
            print(f"      → {detail}")

        _chk(loss_ok,
             f"Progressive val_loss ≤ vanilla × {1+threshold:.2f}",
             f"progressive={prog['val_loss']:.4f}, "
             f"vanilla={ref['val_loss']:.4f}, ratio={loss_ratio:.3f}")
        _chk(speed_ok,
             f"Progressive speed ≥ vanilla × {speed_threshold:.2f}",
             f"progressive={prog.get('steps_per_second',0):.2f} s/s, "
             f"vanilla={ref.get('steps_per_second',0):.2f} s/s, "
             f"ratio={speed_ratio:.3f}")

        overall_go = loss_ok and speed_ok
        verdict = (
            f"{C.GREEN}{C.BOLD}GO ✓ — Progressive drop is viable{C.RESET}"
            if overall_go else
            f"{C.RED}{C.BOLD}NO-GO ✗ — Check pivot strategies in PLAN.md{C.RESET}")
        print(f"\n  {verdict}")

    print(f"{C.BOLD}{'═' * 70}{C.RESET}")


def save_results_json(results, out_dir, args):
    """Save full results dict and args to {out_dir}/results_summary.json."""
    payload = {
        'timestamp': datetime.datetime.now().isoformat(),
        'args': vars(args),
        'results': results,
    }
    path = os.path.join(out_dir, 'results_summary.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    print(f"\n  {C.CYAN}Results saved:{C.RESET} {path}")


def save_results_csv(results, out_dir):
    """Save comparison CSV to {out_dir}/results_summary.csv."""
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
        description='3-Way BERT Comparison: Vanilla vs TokenDrop vs Progressive Drop',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Data
    p.add_argument('--data_path', default=
        '/data/umityilmaz/experiment_1/vanilla-bert/bookcorpus_static_mlm.csv')
    p.add_argument('--output_dir', default=
        '/data/umityilmaz/token_drop_l2/checkpoints/train_csv')
    p.add_argument('--max_samples', type=int, default=0,
                   help='Limit dataset rows (0 = all)')
    p.add_argument('--val_ratio',   type=float, default=0.05)
    p.add_argument('--test_ratio',  type=float, default=0.05)

    # Training
    p.add_argument('--epochs',        type=int,   default=5)
    p.add_argument('--max_steps',     type=int,   default=0,
                   help='Stop after N steps regardless of epoch (0 = epoch-based)')
    p.add_argument('--batch_size',    type=int,   default=64)
    p.add_argument('--learning_rate', type=float, default=1e-4)
    p.add_argument('--log_every',     type=int,   default=100)

    # Regularization / overfitting prevention
    p.add_argument('--weight_decay',            type=float, default=0.01,
                   help='AdamW weight decay coefficient (0 = plain Adam)')
    p.add_argument('--warmup_ratio',            type=float, default=0.05,
                   help='Fraction of total steps used for LR linear warmup')
    p.add_argument('--early_stopping_patience', type=int,   default=5,
                   help='Stop if val_loss does not improve for N epochs (0 = off)')

    # Architecture (BERT-Mini defaults matching experiment_2)
    p.add_argument('--hidden_size',       type=int,   default=256)
    p.add_argument('--num_layers',        type=int,   default=4)
    p.add_argument('--num_heads',         type=int,   default=4)
    p.add_argument('--intermediate_size', type=int,   default=1024)
    p.add_argument('--max_seq_len',       type=int,   default=128)
    p.add_argument('--dropout_rate',      type=float, default=0.1)

    # Token drop budgets (sized for seq_len=64 data)
    p.add_argument('--token_keep_k',  type=int, default=32,
                   help='TokenDrop: keep K most important tokens (of seq_len=64)')
    p.add_argument('--token_keep_k1', type=int, default=48,
                   help='Progressive: tokens after drop-1  (N→k1)')
    p.add_argument('--token_keep_k2', type=int, default=32,
                   help='Progressive: tokens after drop-2  (k1→k2)')
    p.add_argument('--token_keep_k3', type=int, default=16,
                   help='Progressive: tokens after drop-3  (k2→k3)')

    # Control
    p.add_argument('--models', nargs='+',
                   default=['vanilla', 'tokendrop', 'progressive'],
                   choices=['vanilla', 'tokendrop', 'progressive'],
                   help='Which models to train')
    p.add_argument('--go_threshold',    type=float, default=0.10,
                   help='Max allowed loss increase for GO verdict')
    p.add_argument('--speed_threshold', type=float, default=0.85,
                   help='Min speed ratio for GO verdict')
    return p.parse_args()


def main():
    args = parse_args()

    # GPU memory growth — avoids OOM from fragmentation
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    os.makedirs(args.output_dir, exist_ok=True)

    # Global run log
    run_log = os.path.join(args.output_dir, 'run.log')
    run_logger = setup_logger('run', run_log)
    run_logger.info(
        f"Run started | models={args.models} | epochs={args.epochs} | "
        f"batch={args.batch_size} | lr={args.learning_rate} | "
        f"hidden={args.hidden_size} | layers={args.num_layers} | "
        f"k_td={args.token_keep_k} | k1={args.token_keep_k1} | "
        f"k2={args.token_keep_k2} | k3={args.token_keep_k3}")

    print(f"\n{C.CYAN}{C.BOLD}{'─'*65}")
    print(f"  3-Way BERT CSV Comparison")
    print(f"  Vanilla  ·  TokenDrop  ·  Progressive Drop")
    print(f"{'─'*65}{C.RESET}")
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

    # Load data
    train_data, val_data, test_data = load_csv_data(
        args.data_path,
        max_samples=args.max_samples or None,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio)

    train_ds = make_tf_dataset(train_data, args.batch_size, shuffle=True)
    val_ds   = make_tf_dataset(val_data,   args.batch_size, shuffle=False)
    test_ds  = make_tf_dataset(test_data,  args.batch_size, shuffle=False)

    # ── LR schedule parameters (computed once, shared across models) ──────────
    n_train         = len(train_data['input_ids'])
    steps_per_epoch = n_train // args.batch_size
    total_steps     = args.epochs * steps_per_epoch
    if args.max_steps > 0:
        total_steps = min(args.max_steps, total_steps)
    warmup_steps    = int(total_steps * args.warmup_ratio)
    print(f"  LR sched: total_steps={total_steps:,}  "
          f"warmup_steps={warmup_steps:,}  ({args.warmup_ratio:.0%})")

    # Build models
    print(f"\n{C.BOLD}Building models...{C.RESET}")
    all_models = build_models(args)
    model_list = [(n, m, c) for n, m, c in all_models if n in args.models]

    # Train each model sequentially
    results = {}
    for name, model, color in model_list:
        log_dir = os.path.join(args.output_dir, name)
        os.makedirs(log_dir, exist_ok=True)

        results[name] = train_one_model(
            name, model, color, train_ds, val_ds, args, log_dir,
            total_steps=total_steps, warmup_steps=warmup_steps)

        # ── Test set evaluation (using best checkpoint) ───────────────────────
        best_ckpt = os.path.join(log_dir, 'best_model.weights.h5')
        if os.path.exists(best_ckpt):
            model.load_weights(best_ckpt)
            print(f"  {color}Evaluating {name} on test set "
                  f"(best epoch {results[name]['best_epoch']})...{C.RESET}")
        te_losses, te_accs = [], []
        for batch in test_ds:
            tl, ta = eval_step(model, batch)
            te_losses.append(float(tl))
            te_accs.append(float(ta))
        te_loss = sum(te_losses) / max(len(te_losses), 1)
        te_acc  = sum(te_accs)   / max(len(te_accs), 1)
        te_ppl  = math.exp(min(te_loss, 20))
        results[name]['test_loss'] = te_loss
        results[name]['test_acc']  = te_acc
        results[name]['test_ppl']  = te_ppl
        print(f"  {color}[{name}] Test: loss={te_loss:.4f}  "
              f"acc={te_acc:.4f}  ppl={te_ppl:.2f}{C.RESET}")

        # ── Forward-pass latency measurement ──────────────────────────────────
        print(f"  {color}Measuring {name} forward-pass latency...{C.RESET}")
        sample_batch = next(iter(train_ds))
        # Warmup (3 runs)
        for _ in range(3):
            _out = model(sample_batch, training=False)
            tf.reduce_sum(_out).numpy()
        # Timed runs (10 runs)
        _lat_times = []
        for _ in range(10):
            _t0 = time.perf_counter()
            _out = model(sample_batch, training=False)
            tf.reduce_sum(_out).numpy()
            _lat_times.append((time.perf_counter() - _t0) * 1000)
        latency_ms = sum(_lat_times) / len(_lat_times)
        results[name]['forward_latency_ms'] = round(latency_ms, 3)
        print(f"  {color}[{name}] Latency: {latency_ms:.2f} ms / batch{C.RESET}")

        run_logger.info(
            f"[{name}] finished | val_loss={results[name]['val_loss']:.4f} | "
            f"val_acc={results[name]['val_acc']:.4f} | "
            f"test_loss={te_loss:.4f} | test_acc={te_acc:.4f} | "
            f"best_epoch={results[name]['best_epoch']}")

    # ── Final comparison + persist results ────────────────────────────────────
    print_comparison(results,
                     threshold=args.go_threshold,
                     speed_threshold=args.speed_threshold)
    save_results_json(results, args.output_dir, args)
    save_results_csv(results, args.output_dir)

    run_logger.info("Run complete.")

    # TensorBoard hint
    print(f"\n  {C.CYAN}TensorBoard:{C.RESET}")
    print(f"    tensorboard --logdir {args.output_dir} --port 6006")
    print(f"\n  {C.CYAN}Log files:{C.RESET}")
    for name in results:
        log_dir = os.path.join(args.output_dir, name)
        print(f"    {name:12s}: {log_dir}/training-{name}.log")
    print()


if __name__ == '__main__':
    main()
