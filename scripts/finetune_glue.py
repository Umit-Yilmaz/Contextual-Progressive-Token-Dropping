#!/usr/bin/env python3
# =============================================================================
# finetune_glue.py  —  Multi-task GLUE Fine-tuning from Pretrained Checkpoints
#
# Fine-tunes pretrained BERT encoders on 4 GLUE tasks for Go/No-Go #4:
#   CoLA  — single sentence, binary, metric: MCC
#   MRPC  — sentence pair, binary, metric: F1
#   STS-B — sentence pair, regression (0–5), metric: Pearson
#   RTE   — sentence pair, binary, metric: Accuracy
#
# Runs 3 models × 4 tasks = 12 fine-tuning runs sequentially.
#
# Usage:
#   python scripts/finetune_glue.py \
#     --checkpoint_dir /data/umityilmaz/token_drop_l2/checkpoints/train_base512 \
#     --output_dir /data/umityilmaz/token_drop_l2/checkpoints/finetune_glue \
#     --epochs 5 --batch_size 32 --learning_rate 2e-5 \
#     --models vanilla tokendrop progressive
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
warnings.filterwarnings('ignore', category=UserWarning)

# ── Repo path ─────────────────────────────────────────────────────────────────
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'experiments', 'progressive_contextual_dropping'))
sys.path.insert(0, REPO)

import numpy as np
import tensorflow as tf

VOCAB_SIZE = 30522  # bert-base-uncased


# ── Task definitions ──────────────────────────────────────────────────────────

TASK_CONFIG = {
    'cola': {
        'glue_name':     'cola',
        'sentence_keys': ('sentence',),
        'num_classes':   2,
        'is_regression': False,
        'metric_name':   'mcc',
        'metric_higher_better': True,
    },
    'mrpc': {
        'glue_name':     'mrpc',
        'sentence_keys': ('sentence1', 'sentence2'),
        'num_classes':   2,
        'is_regression': False,
        'metric_name':   'f1',
        'metric_higher_better': True,
    },
    'stsb': {
        'glue_name':     'stsb',
        'sentence_keys': ('sentence1', 'sentence2'),
        'num_classes':   1,
        'is_regression': True,
        'metric_name':   'pearson',
        'metric_higher_better': True,
    },
    'rte': {
        'glue_name':     'rte',
        'sentence_keys': ('sentence1', 'sentence2'),
        'num_classes':   2,
        'is_regression': False,
        'metric_name':   'accuracy',
        'metric_higher_better': True,
    },
}


# ── ANSI colours ──────────────────────────────────────────────────────────────
class C:
    BOLD   = '\033[1m'
    RESET  = '\033[0m'
    BLUE   = '\033[94m'
    YELLOW = '\033[93m'
    GREEN  = '\033[92m'
    RED    = '\033[91m'
    CYAN   = '\033[96m'
    MAGENTA = '\033[95m'


# ── LR Schedule ───────────────────────────────────────────────────────────────

class WarmupLinearDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, peak_lr, warmup_steps, total_steps):
        super().__init__()
        self.peak_lr      = float(peak_lr)
        self.warmup_steps = float(max(warmup_steps, 1))
        self.total_steps  = float(max(total_steps, warmup_steps + 1))

    def __call__(self, step):
        step      = tf.cast(step, tf.float32)
        warmup_lr = self.peak_lr * step / self.warmup_steps
        decay_lr  = self.peak_lr * tf.maximum(
            0.0, (self.total_steps - step) /
            tf.maximum(self.total_steps - self.warmup_steps, 1.0))
        return tf.where(step < self.warmup_steps, warmup_lr, decay_lr)

    def get_config(self):
        return {'peak_lr': self.peak_lr, 'warmup_steps': self.warmup_steps,
                'total_steps': self.total_steps}


# ── BertMLM (for weight loading) ─────────────────────────────────────────────

class BertMLM(tf.keras.Model):
    """BERT encoder + MLM head. Used only to load pretrained weights."""
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
        h       = self.mlm_dense(enc_out['sequence_output'])
        h       = tf.nn.gelu(h, approximate=True)
        h       = self.mlm_norm(h)
        return self.mlm_proj(h)


# ── BertClassifier (for classification tasks) ────────────────────────────────

class BertClassifier(tf.keras.Model):
    """BERT encoder + classification head (pooled [CLS] → Dense)."""
    def __init__(self, encoder, num_classes, dropout_rate=0.1,
                 name='bert_classifier'):
        super().__init__(name=name)
        self.encoder    = encoder
        self.dropout    = tf.keras.layers.Dropout(dropout_rate)
        self.classifier = tf.keras.layers.Dense(num_classes,
                                                name='classifier')

    def call(self, inputs, training=False):
        enc_inp = {
            'input_word_ids': inputs['input_ids'],
            'input_mask':     inputs['attention_mask'],
            'input_type_ids': tf.zeros_like(inputs['input_ids']),
        }
        enc_out = self.encoder(enc_inp, training=training)
        pooled = enc_out['pooled_output']
        pooled = self.dropout(pooled, training=training)
        return self.classifier(pooled)


# ── BertRegressor (for STS-B) ────────────────────────────────────────────────

class BertRegressor(tf.keras.Model):
    """BERT encoder + regression head (pooled [CLS] → Dense(1))."""
    def __init__(self, encoder, dropout_rate=0.1, name='bert_regressor'):
        super().__init__(name=name)
        self.encoder   = encoder
        self.dropout   = tf.keras.layers.Dropout(dropout_rate)
        self.regressor = tf.keras.layers.Dense(1, name='regressor')

    def call(self, inputs, training=False):
        enc_inp = {
            'input_word_ids': inputs['input_ids'],
            'input_mask':     inputs['attention_mask'],
            'input_type_ids': tf.zeros_like(inputs['input_ids']),
        }
        enc_out = self.encoder(enc_inp, training=training)
        pooled = enc_out['pooled_output']
        pooled = self.dropout(pooled, training=training)
        return tf.squeeze(self.regressor(pooled), axis=-1)  # [B]


# ── Metrics ───────────────────────────────────────────────────────────────────

def matthews_corrcoef(y_true, y_pred):
    """Compute Matthews Correlation Coefficient."""
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    denom = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    if denom == 0:
        return 0.0
    return (tp*tn - fp*fn) / denom


def f1_score(y_true, y_pred):
    """Compute F1 score for binary classification."""
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def pearson_corr(y_true, y_pred):
    """Compute Pearson correlation coefficient."""
    if len(y_true) < 2:
        return 0.0
    mean_t = np.mean(y_true)
    mean_p = np.mean(y_pred)
    num = np.sum((y_true - mean_t) * (y_pred - mean_p))
    den = math.sqrt(np.sum((y_true - mean_t)**2) * np.sum((y_pred - mean_p)**2))
    if den == 0:
        return 0.0
    return num / den


def compute_metric(task_name, y_true, y_pred):
    """Compute task-specific metric. Returns (metric_value, metric_name)."""
    cfg = TASK_CONFIG[task_name]
    if cfg['is_regression']:
        return pearson_corr(y_true, y_pred), 'pearson'
    elif cfg['metric_name'] == 'mcc':
        preds = (y_pred > 0.5).astype(np.int32) if y_pred.dtype == np.float32 else y_pred
        return matthews_corrcoef(y_true, preds), 'mcc'
    elif cfg['metric_name'] == 'f1':
        return f1_score(y_true, y_pred), 'f1'
    else:  # accuracy
        return np.mean(y_true == y_pred), 'accuracy'


# ── Data loading ──────────────────────────────────────────────────────────────

def load_glue_task(task_name, max_seq_len, cache_dir=None):
    """Load a GLUE task from HuggingFace and tokenize.

    Returns:
      train_data, val_data : dicts with 'input_ids', 'attention_mask', 'labels'
    """
    from datasets import load_dataset
    from transformers import BertTokenizerFast

    cfg = TASK_CONFIG[task_name]
    print(f"\n{C.CYAN}Loading GLUE/{task_name} dataset...{C.RESET}")

    ds = load_dataset('glue', cfg['glue_name'], cache_dir=cache_dir)
    tokenizer = BertTokenizerFast.from_pretrained(
        'bert-base-uncased', cache_dir=cache_dir)

    sent_keys = cfg['sentence_keys']

    def tokenize_fn(examples):
        if len(sent_keys) == 1:
            return tokenizer(examples[sent_keys[0]],
                             max_length=max_seq_len,
                             padding='max_length',
                             truncation=True)
        else:
            return tokenizer(examples[sent_keys[0]],
                             examples[sent_keys[1]],
                             max_length=max_seq_len,
                             padding='max_length',
                             truncation=True)

    # Columns to remove
    remove_cols = list(sent_keys) + ['idx']
    remove_cols = [c for c in remove_cols if c in ds['train'].column_names]

    train_ds = ds['train'].map(tokenize_fn, batched=True,
                                remove_columns=remove_cols)
    val_key = 'validation' if 'validation' in ds else 'validation_matched'
    val_ds   = ds[val_key].map(tokenize_fn, batched=True,
                                remove_columns=remove_cols)

    label_dtype = np.float32 if cfg['is_regression'] else np.int32

    def to_numpy(split):
        return {
            'input_ids':      np.array(split['input_ids'],      dtype=np.int32),
            'attention_mask': np.array(split['attention_mask'],  dtype=np.int32),
            'labels':         np.array(split['label'],           dtype=label_dtype),
        }

    train_data = to_numpy(train_ds)
    val_data   = to_numpy(val_ds)

    n_train = len(train_data['labels'])
    n_val   = len(val_data['labels'])
    print(f"  Train: {n_train:,} | Val: {n_val:,} | Seq: {max_seq_len} | "
          f"Type: {'regression' if cfg['is_regression'] else 'classification'} | "
          f"Metric: {cfg['metric_name']}")
    return train_data, val_data


def make_tf_dataset(data, batch_size, shuffle=True, seed=42):
    ds = tf.data.Dataset.from_tensor_slices(data)
    if shuffle:
        ds = ds.shuffle(buffer_size=min(10_000, len(data['input_ids'])),
                        seed=seed)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ── Encoder + weight loading ─────────────────────────────────────────────────

def build_and_load_encoder(model_name, args):
    """Build encoder, wrap in BertMLM, load pretrained weights, return encoder."""
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

    ft_seq = args.finetune_seq_len
    ft_k   = ft_seq // 2
    ft_k1  = int(ft_seq * 0.75)
    ft_k2  = ft_seq // 2
    ft_k3  = ft_seq // 4

    if model_name == 'vanilla':
        encoder = _bert_enc_mod.BertEncoder(
            vocab_size          = VOCAB_SIZE,
            inner_dim           = args.intermediate_size,
            max_sequence_length = args.pretrain_seq_len,
            dict_outputs        = True,
            **common)
        mlm_name = 'vanilla_mlm'

    elif model_name == 'tokendrop':
        encoder = TokenDropBertEncoder(
            vocab_size          = VOCAB_SIZE,
            intermediate_size   = args.intermediate_size,
            max_sequence_length = args.pretrain_seq_len,
            token_keep_k        = ft_k,
            **common)
        mlm_name = 'tokendrop_mlm'

    elif model_name == 'progressive':
        encoder = ProgressiveContextualDropEncoder(
            vocab_size          = VOCAB_SIZE,
            inner_dim           = args.intermediate_size,
            max_sequence_length = args.pretrain_seq_len,
            token_keep_k1       = ft_k1,
            token_keep_k2       = ft_k2,
            token_keep_k3       = ft_k3,
            **common)
        mlm_name = 'progressive_mlm'
    else:
        raise ValueError(f'Unknown model: {model_name}')

    mlm_model = BertMLM(encoder, VOCAB_SIZE, args.hidden_size, name=mlm_name)

    print(f"  Building {model_name} model (warm-up)...", flush=True)
    dummy = {
        'input_ids':      tf.zeros([2, ft_seq], dtype=tf.int32),
        'attention_mask':  tf.ones([2, ft_seq], dtype=tf.int32),
    }
    _ = mlm_model(dummy, training=False)
    n_params = sum(int(tf.size(v).numpy()) for v in mlm_model.trainable_variables)
    print(f"    Total params (MLM): {n_params:,}")

    ckpt_path = os.path.join(
        args.checkpoint_dir, model_name, 'best_model.weights.h5')
    print(f"  Loading weights: {ckpt_path}")
    mlm_model.load_weights(ckpt_path)
    print(f"    {C.GREEN}Weights loaded successfully.{C.RESET}")

    return encoder


# ── Logging ───────────────────────────────────────────────────────────────────

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


# ── Training / eval steps (classification) ───────────────────────────────────

@tf.function
def train_step_cls(model, optimizer, batch):
    with tf.GradientTape() as tape:
        logits = model(batch, training=True)
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            batch['labels'], logits, from_logits=True)
        loss = tf.reduce_mean(loss)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
    acc   = tf.reduce_mean(tf.cast(tf.equal(preds, batch['labels']), tf.float32))
    return loss, acc, preds


@tf.function
def eval_step_cls(model, batch):
    logits = model(batch, training=False)
    loss = tf.keras.losses.sparse_categorical_crossentropy(
        batch['labels'], logits, from_logits=True)
    loss = tf.reduce_mean(loss)
    preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
    acc   = tf.reduce_mean(tf.cast(tf.equal(preds, batch['labels']), tf.float32))
    return loss, acc, preds


# ── Training / eval steps (regression — STS-B) ──────────────────────────────

@tf.function
def train_step_reg(model, optimizer, batch):
    with tf.GradientTape() as tape:
        preds = model(batch, training=True)  # [B]
        loss = tf.reduce_mean(tf.square(preds - batch['labels']))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, preds


@tf.function
def eval_step_reg(model, batch):
    preds = model(batch, training=False)
    loss = tf.reduce_mean(tf.square(preds - batch['labels']))
    return loss, preds


# ── Fine-tune one model on one task ──────────────────────────────────────────

def finetune_one(model_name, task_name, encoder, color, train_ds, val_ds,
                 val_data_labels, args, log_dir, total_steps, warmup_steps):
    """Fine-tune a pretrained encoder on one GLUE task.

    Returns dict with best metric, loss, etc.
    """
    cfg = TASK_CONFIG[task_name]
    is_reg = cfg['is_regression']
    metric_name = cfg['metric_name']

    print(f"\n{C.BOLD}{'═' * 65}{C.RESET}")
    print(f"  {color}{C.BOLD}▶  {model_name.upper()} × {task_name.upper()} "
          f"({'regression' if is_reg else 'classification'}, "
          f"metric={metric_name}){C.RESET}")
    print(f"{C.BOLD}{'═' * 65}{C.RESET}")

    # Build task head
    if is_reg:
        head = BertRegressor(encoder, dropout_rate=args.dropout_rate,
                             name=f'{model_name}_{task_name}_reg')
    else:
        head = BertClassifier(encoder, num_classes=cfg['num_classes'],
                              dropout_rate=args.dropout_rate,
                              name=f'{model_name}_{task_name}_cls')

    # Warm up
    dummy = {
        'input_ids':      tf.zeros([2, args.finetune_seq_len], dtype=tf.int32),
        'attention_mask':  tf.ones([2, args.finetune_seq_len], dtype=tf.int32),
        'labels':         tf.zeros([2], dtype=tf.float32 if is_reg else tf.int32),
    }
    _ = head(dummy, training=False)

    cls_params = sum(int(tf.size(v).numpy()) for v in head.trainable_variables)
    enc_params = sum(int(tf.size(v).numpy()) for v in encoder.trainable_variables)
    print(f"  Encoder: {enc_params:,} | Head: {cls_params - enc_params:,} | "
          f"Total: {cls_params:,}")

    # Logger
    log_path = os.path.join(log_dir, f'finetune-{model_name}-{task_name}.log')
    csv_path = os.path.join(log_dir, f'finetune-{model_name}-{task_name}.csv')
    logger = setup_logger(f'ft_{model_name}_{task_name}', log_path)
    logger.info(f"Fine-tune started | model={model_name} | task={task_name} | "
                f"epochs={args.epochs} | batch={args.batch_size} | "
                f"lr={args.learning_rate}")

    # Optimizer
    lr_schedule = WarmupLinearDecay(
        peak_lr=args.learning_rate,
        warmup_steps=warmup_steps,
        total_steps=max(total_steps, 1))
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=args.weight_decay,
        beta_1=0.9, beta_2=0.999, epsilon=1e-8)

    # Eager warm-up (build optimizer state)
    _wb = next(iter(train_ds))
    with tf.GradientTape() as _t:
        if is_reg:
            _lo = head(_wb, training=True)
            _ls = tf.reduce_mean(tf.square(_lo - _wb['labels']))
        else:
            _lo = head(_wb, training=True)
            _ls = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
                _wb['labels'], _lo, from_logits=True))
    _g = _t.gradient(_ls, head.trainable_variables)
    optimizer.apply_gradients(zip(_g, head.trainable_variables))
    optimizer.iterations.assign(0)
    del _wb, _lo, _ls, _g, _t

    best_metric = -1e9 if cfg['metric_higher_better'] else 1e9
    best_epoch  = 0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        # ── Training ─────────────────────────────────────────────────────
        tr_losses = []
        t0 = time.time()

        for batch in train_ds:
            if is_reg:
                loss, _ = train_step_reg(head, optimizer, batch)
            else:
                loss, _, _ = train_step_cls(head, optimizer, batch)
            tr_losses.append(float(loss))
            global_step += 1

        elapsed = time.time() - t0
        n_steps = max(len(tr_losses), 1)
        sps     = n_steps / max(elapsed, 1e-6)

        # ── Validation (collect all predictions for metric) ───────────
        va_losses = []
        all_preds = []

        for batch in val_ds:
            if is_reg:
                vl, vp = eval_step_reg(head, batch)
                all_preds.append(vp.numpy())
            else:
                vl, _, vp = eval_step_cls(head, batch)
                all_preds.append(vp.numpy())
            va_losses.append(float(vl))

        t_loss = sum(tr_losses) / n_steps
        v_loss = sum(va_losses) / max(len(va_losses), 1)

        # Compute task-specific metric on full val set
        all_preds_np = np.concatenate(all_preds)
        # Truncate to match labels (drop_remainder may have truncated)
        n_eval = min(len(all_preds_np), len(val_data_labels))
        metric_val, _ = compute_metric(
            task_name, val_data_labels[:n_eval], all_preds_np[:n_eval])

        lr_now = float(lr_schedule(optimizer.iterations))

        # Log
        logger.info(
            f"Epoch {epoch}/{args.epochs} | LR: {lr_now:.3e} | "
            f"Train loss={t_loss:.4f} | Val loss={v_loss:.4f} | "
            f"{metric_name}={metric_val:.4f} | {sps:.1f} step/s")

        # CSV
        write_header = not os.path.exists(csv_path)
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=[
                'epoch', 'global_step', 'lr', 'train_loss',
                'val_loss', metric_name, 'steps_per_second', 'elapsed_s'])
            if write_header:
                w.writeheader()
            w.writerow({
                'epoch': epoch, 'global_step': global_step,
                'lr': round(lr_now, 8),
                'train_loss': round(t_loss, 6),
                'val_loss': round(v_loss, 6),
                metric_name: round(metric_val, 6),
                'steps_per_second': round(sps, 4),
                'elapsed_s': round(elapsed, 2),
            })

        # Console
        print(f"  {color}[{model_name}×{task_name}]{C.RESET}"
              f"  Epoch {epoch:2d}/{args.epochs}"
              f"  | LR: {lr_now:.2e}"
              f"  | T_loss={t_loss:.4f}"
              f"  | V_loss={v_loss:.4f}"
              f"  | {metric_name}={metric_val:.4f}"
              f"  | {sps:.1f} step/s")

        # Best checkpoint
        is_better = (metric_val > best_metric if cfg['metric_higher_better']
                     else metric_val < best_metric)
        if is_better:
            best_metric = metric_val
            best_epoch  = epoch
            head.save_weights(
                os.path.join(log_dir, f'best-{model_name}-{task_name}.weights.h5'))
            print(f"    {C.GREEN}↑ New best {metric_name}={metric_val:.4f} "
                  f"(epoch {epoch}){C.RESET}")

    logger.info(f"Complete! Best {metric_name}={best_metric:.4f} "
                f"at epoch {best_epoch}")

    return {
        'model': model_name,
        'task': task_name,
        f'best_{metric_name}': best_metric,
        'best_epoch': best_epoch,
        'final_val_loss': v_loss,
        'final_train_loss': t_loss,
        'steps_per_second': sps,
        'global_step': global_step,
    }


# ── Comparison and Go/No-Go ──────────────────────────────────────────────────

def print_task_results(task_results, task_name):
    """Print results for one GLUE task."""
    cfg = TASK_CONFIG[task_name]
    mn  = cfg['metric_name']

    print(f"\n  {C.BOLD}── {task_name.upper()} ({mn}) ──{C.RESET}")
    color_map = {'vanilla': C.BLUE, 'tokendrop': C.YELLOW,
                 'progressive': C.GREEN}
    for r in task_results:
        c = color_map.get(r['model'], '')
        val = r.get(f'best_{mn}', 0)
        print(f"    {c}{r['model']:<14}{C.RESET}"
              f"  {mn}={val:.4f}"
              f"  epoch={r['best_epoch']}"
              f"  {r['steps_per_second']:.1f} step/s")


def print_final_comparison(all_results, model_names):
    """Print cross-task comparison and Go/No-Go #4."""
    print(f"\n{C.BOLD}{'═' * 75}{C.RESET}")
    print(f"{C.BOLD}  GLUE Fine-tuning Results — Go/No-Go #4{C.RESET}")
    print(f"{C.BOLD}{'═' * 75}{C.RESET}")

    tasks = list(TASK_CONFIG.keys())
    color_map = {'vanilla': C.BLUE, 'tokendrop': C.YELLOW,
                 'progressive': C.GREEN}

    # Header
    hdr = f"  {'Model':<14}"
    for t in tasks:
        mn = TASK_CONFIG[t]['metric_name']
        hdr += f"  {t}({mn})"
    hdr += "   Avg"
    print(hdr)
    print(f"  {'─'*14}" + "".join(f"  {'─'*12}" for _ in tasks) + f"  {'─'*8}")

    # Rows
    model_avgs = {}
    for m in model_names:
        c = color_map.get(m, '')
        row = f"  {c}{m:<14}{C.RESET}"
        scores = []
        for t in tasks:
            mn = TASK_CONFIG[t]['metric_name']
            r = next((x for x in all_results
                      if x['model'] == m and x['task'] == t), None)
            if r:
                val = r.get(f'best_{mn}', 0)
                scores.append(val)
                row += f"  {val:>12.4f}"
            else:
                row += f"  {'—':>12}"
        avg = sum(scores) / max(len(scores), 1)
        model_avgs[m] = avg
        row += f"  {avg:>8.4f}"
        print(row)

    # Go/No-Go #4: ProgDrop ≥ vanilla on ≥3/4 tasks
    van_results = [x for x in all_results if x['model'] == 'vanilla']
    prog_results = [x for x in all_results if x['model'] == 'progressive']

    prog_wins = 0
    print(f"\n  {C.BOLD}Go/No-Go #4 — Per-task comparison (Progressive vs Vanilla):{C.RESET}")
    for t in tasks:
        mn = TASK_CONFIG[t]['metric_name']
        van_r  = next((x for x in van_results if x['task'] == t), None)
        prog_r = next((x for x in prog_results if x['task'] == t), None)
        if van_r and prog_r:
            van_val  = van_r.get(f'best_{mn}', 0)
            prog_val = prog_r.get(f'best_{mn}', 0)
            delta = prog_val - van_val
            win = delta >= 0
            if win:
                prog_wins += 1
            sym = f"{C.GREEN}✓{C.RESET}" if win else f"{C.RED}✗{C.RESET}"
            print(f"    {sym}  {t:<6} {mn}:  vanilla={van_val:.4f}  "
                  f"progressive={prog_val:.4f}  Δ={delta:+.4f}")

    # Verdict
    go = prog_wins >= 3
    print(f"\n  Progressive wins: {prog_wins}/4 tasks (threshold: ≥3)")
    if go:
        print(f"\n  {C.GREEN}{C.BOLD}✅ GO #4 — Progressive shows consistent "
              f"downstream advantage{C.RESET}")
    else:
        print(f"\n  {C.RED}{C.BOLD}❌ NO-GO #4 — Progressive does not show "
              f"consistent advantage{C.RESET}")

    # Avg comparison
    if 'vanilla' in model_avgs and 'progressive' in model_avgs:
        delta_avg = model_avgs['progressive'] - model_avgs['vanilla']
        print(f"\n  Average metric: vanilla={model_avgs['vanilla']:.4f}  "
              f"progressive={model_avgs['progressive']:.4f}  "
              f"Δ={delta_avg:+.4f}")

    print(f"{C.BOLD}{'═' * 75}{C.RESET}")
    return go


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Multi-task GLUE Fine-tuning for Go/No-Go #4',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('--checkpoint_dir',
        default='/data/umityilmaz/token_drop_l2/checkpoints/train_base512')
    p.add_argument('--output_dir',
        default='/data/umityilmaz/token_drop_l2/checkpoints/finetune_glue')
    p.add_argument('--cache_dir',
        default='/data/umityilmaz/token_drop_l2/data/hf_cache')

    p.add_argument('--epochs',        type=int,   default=5)
    p.add_argument('--batch_size',    type=int,   default=32)
    p.add_argument('--learning_rate', type=float, default=2e-5)
    p.add_argument('--warmup_ratio',  type=float, default=0.1)
    p.add_argument('--weight_decay',  type=float, default=0.01)
    p.add_argument('--dropout_rate',  type=float, default=0.1)

    p.add_argument('--hidden_size',       type=int, default=768)
    p.add_argument('--num_layers',        type=int, default=12)
    p.add_argument('--num_heads',         type=int, default=12)
    p.add_argument('--intermediate_size', type=int, default=3072)
    p.add_argument('--pretrain_seq_len',  type=int, default=512)
    p.add_argument('--finetune_seq_len',  type=int, default=128)

    p.add_argument('--models', nargs='+',
                   default=['vanilla', 'tokendrop', 'progressive'],
                   choices=['vanilla', 'tokendrop', 'progressive'])
    p.add_argument('--tasks', nargs='+',
                   default=['cola', 'mrpc', 'stsb', 'rte'],
                   choices=['cola', 'mrpc', 'stsb', 'rte'])
    p.add_argument('--seed', type=int, default=42)

    return p.parse_args()


def main():
    args = parse_args()

    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Config ────────────────────────────────────────────────────────────────
    print(f"\n{C.CYAN}{C.BOLD}{'─'*65}")
    print(f"  GLUE Fine-tuning — Go/No-Go #4 Downstream Evaluation")
    print(f"{'─'*65}{C.RESET}")
    print(f"  Checkpoint  : {args.checkpoint_dir}")
    print(f"  Output      : {args.output_dir}")
    print(f"  Models      : {', '.join(args.models)}")
    print(f"  Tasks       : {', '.join(args.tasks)}")
    print(f"  Architecture: hidden={args.hidden_size}, layers={args.num_layers}, "
          f"heads={args.num_heads}")
    print(f"  Pretrain seq: {args.pretrain_seq_len}  |  Fine-tune seq: "
          f"{args.finetune_seq_len}")
    ft_seq = args.finetune_seq_len
    print(f"  Token budgets: tokendrop k={ft_seq//2}, "
          f"progressive k1={int(ft_seq*0.75)}/k2={ft_seq//2}/k3={ft_seq//4}")
    print(f"  Training    : epochs={args.epochs}, batch={args.batch_size}, "
          f"lr={args.learning_rate}")

    all_results = []
    color_map = {'vanilla': C.BLUE, 'tokendrop': C.YELLOW,
                 'progressive': C.GREEN}

    # ── Loop: task → model (reload encoder fresh per model×task) ─────────
    for task_name in args.tasks:
        task_cfg = TASK_CONFIG[task_name]

        # Load data once per task
        train_data, val_data = load_glue_task(
            task_name, max_seq_len=args.finetune_seq_len,
            cache_dir=args.cache_dir)

        train_ds = make_tf_dataset(train_data, args.batch_size,
                                    shuffle=True, seed=args.seed)
        val_ds   = make_tf_dataset(val_data, args.batch_size, shuffle=False)

        n_train         = len(train_data['input_ids'])
        steps_per_epoch = n_train // args.batch_size
        total_steps     = args.epochs * steps_per_epoch
        warmup_steps    = int(total_steps * args.warmup_ratio)
        print(f"  LR: total_steps={total_steps:,}  warmup={warmup_steps:,}")

        for model_name in args.models:
            color   = color_map.get(model_name, '')
            log_dir = os.path.join(args.output_dir, f'{task_name}_{model_name}')
            os.makedirs(log_dir, exist_ok=True)

            # Build fresh encoder + load pretrained weights
            encoder = build_and_load_encoder(model_name, args)

            result = finetune_one(
                model_name, task_name, encoder, color,
                train_ds, val_ds, val_data['labels'],
                args, log_dir,
                total_steps=total_steps, warmup_steps=warmup_steps)

            all_results.append(result)

            # Print per-task results so far
            task_results = [r for r in all_results if r['task'] == task_name]
            print_task_results(task_results, task_name)

            # Clear session to free GPU
            tf.keras.backend.clear_session()

    # ── Final comparison ──────────────────────────────────────────────────
    go = print_final_comparison(all_results, args.models)

    # Save results
    payload = {
        'timestamp': datetime.datetime.now().isoformat(),
        'args': vars(args),
        'results': all_results,
        'go_no_go_4': go,
    }
    json_path = os.path.join(args.output_dir, 'glue_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    print(f"\n  {C.CYAN}Results:{C.RESET} {json_path}")

    # CSV summary
    csv_path = os.path.join(args.output_dir, 'glue_results.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['model', 'task', 'metric', 'best_value', 'best_epoch',
                      'final_val_loss', 'steps_per_second']
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_results:
            t = r['task']
            mn = TASK_CONFIG[t]['metric_name']
            w.writerow({
                'model': r['model'], 'task': t, 'metric': mn,
                'best_value': r.get(f'best_{mn}', 0),
                'best_epoch': r['best_epoch'],
                'final_val_loss': r['final_val_loss'],
                'steps_per_second': r['steps_per_second'],
            })
    print(f"  {C.CYAN}CSV:{C.RESET}     {csv_path}")


if __name__ == '__main__':
    main()
