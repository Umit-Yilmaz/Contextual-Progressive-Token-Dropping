#!/usr/bin/env python3
# =============================================================================
# finetune_sst2.py  —  SST-2 Fine-tuning from Pretrained MLM Checkpoints
#
# Loads pretrained BERT encoder weights from MLM pretraining checkpoints
# and fine-tunes on SST-2 (binary sentiment classification) for Go/No-Go
# downstream task evaluation.
#
# Weight loading strategy:
#   1. Build BertMLM with SAME architecture + model name as pretraining
#   2. Warm up (forward pass to trigger Keras lazy weight creation)
#   3. Load pretrained .weights.h5
#   4. Extract encoder → build BertClassifier
#   5. Fine-tune all parameters on SST-2
#
# Token dropping during fine-tune:
#   k values are scaled proportionally to finetune_seq_len (default 128):
#     progressive: k1=96, k2=64, k3=32  (same 75/50/25% ratios)
#     tokendrop:   k=64                  (same 50% ratio)
#
# Usage:
#   python scripts/finetune_sst2.py \
#     --checkpoint_dir /data/umityilmaz/token_drop_l2/checkpoints/train_base512 \
#     --output_dir /data/umityilmaz/token_drop_l2/checkpoints/finetune_sst2 \
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


# ── ANSI colours ──────────────────────────────────────────────────────────────
class C:
    BOLD   = '\033[1m'
    RESET  = '\033[0m'
    BLUE   = '\033[94m'
    YELLOW = '\033[93m'
    GREEN  = '\033[92m'
    RED    = '\033[91m'
    CYAN   = '\033[96m'


# ── LR Schedule: linear warmup → linear decay to 0 ───────────────────────────

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


# ── BertMLM (same as train_csv_comparison — needed for weight loading) ───────

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


# ── BertClassifier ────────────────────────────────────────────────────────────

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
        # Use pooled_output (tanh-transformed [CLS]) — better for classification
        pooled = enc_out['pooled_output']                   # [B, H]
        pooled = self.dropout(pooled, training=training)
        return self.classifier(pooled)                      # [B, num_classes]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_sst2_data(max_seq_len, cache_dir=None):
    """Load SST-2 from HuggingFace and tokenize with bert-base-uncased.

    Returns:
      train_data, val_data : dicts with 'input_ids', 'attention_mask', 'labels'
    """
    from datasets import load_dataset
    from transformers import BertTokenizerFast

    print(f"\n{C.CYAN}Loading SST-2 dataset...{C.RESET}")

    ds = load_dataset('glue', 'sst2', cache_dir=cache_dir)
    tokenizer = BertTokenizerFast.from_pretrained(
        'bert-base-uncased', cache_dir=cache_dir)

    def tokenize_fn(examples):
        return tokenizer(examples['sentence'],
                         max_length=max_seq_len,
                         padding='max_length',
                         truncation=True)

    # Tokenize
    train_ds = ds['train'].map(tokenize_fn, batched=True,
                                remove_columns=['sentence', 'idx'])
    val_ds   = ds['validation'].map(tokenize_fn, batched=True,
                                     remove_columns=['sentence', 'idx'])

    def to_numpy(split):
        return {
            'input_ids':      np.array(split['input_ids'],      dtype=np.int32),
            'attention_mask': np.array(split['attention_mask'],  dtype=np.int32),
            'labels':         np.array(split['label'],           dtype=np.int32),
        }

    train_data = to_numpy(train_ds)
    val_data   = to_numpy(val_ds)

    print(f"  Train samples : {len(train_data['labels']):,}")
    print(f"  Val samples   : {len(val_data['labels']):,}")
    print(f"  Seq length    : {max_seq_len}")
    print(f"  Num classes   : 2 (positive/negative)")
    return train_data, val_data


def make_tf_dataset(data, batch_size, shuffle=True, seed=42):
    ds = tf.data.Dataset.from_tensor_slices(data)
    if shuffle:
        ds = ds.shuffle(buffer_size=min(10_000, len(data['input_ids'])),
                        seed=seed)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ── Encoder + weight loading factory ─────────────────────────────────────────

def build_and_load_encoder(model_name, args):
    """Build encoder, wrap in BertMLM, load pretrained weights, return encoder.

    Weight loading requires:
    1. Same encoder architecture (hidden_size, num_layers, etc.)
    2. Same BertMLM wrapper name (vanilla_mlm, tokendrop_mlm, progressive_mlm)
    3. max_sequence_length=pretrain_seq_len (for positional embedding shape)
    4. k values can differ (they don't affect weight shapes)
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

    # Fine-tune k values (scaled proportionally from pretraining)
    ft_seq = args.finetune_seq_len
    ft_k   = ft_seq // 2           # tokendrop: 50%
    ft_k1  = int(ft_seq * 0.75)    # progressive stage 1: 75%
    ft_k2  = ft_seq // 2           # progressive stage 2: 50%
    ft_k3  = ft_seq // 4           # progressive stage 3: 25%

    # Build encoder with pretrain_seq_len for positional embeddings
    # but fine-tune k values for shorter sequences
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

    # Wrap in BertMLM (same name as pretraining for weight path matching)
    mlm_model = BertMLM(encoder, VOCAB_SIZE, args.hidden_size, name=mlm_name)

    # Eager warm-up: build all weights with a forward pass
    print(f"  Building {model_name} model (warm-up)...", flush=True)
    dummy = {
        'input_ids':      tf.zeros([2, ft_seq], dtype=tf.int32),
        'attention_mask':  tf.ones([2, ft_seq], dtype=tf.int32),
    }
    _ = mlm_model(dummy, training=False)
    n_params = sum(int(tf.size(v).numpy()) for v in mlm_model.trainable_variables)
    print(f"    Total params (MLM): {n_params:,}")

    # Load pretrained weights
    ckpt_path = os.path.join(
        args.checkpoint_dir, model_name, 'best_model.weights.h5')
    print(f"  Loading weights: {ckpt_path}")
    mlm_model.load_weights(ckpt_path)
    print(f"    {C.GREEN}Weights loaded successfully.{C.RESET}")

    return encoder


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


# ── Training / eval steps ─────────────────────────────────────────────────────

@tf.function
def train_step(model, optimizer, batch):
    with tf.GradientTape() as tape:
        logits = model(batch, training=True)
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            batch['labels'], logits, from_logits=True)
        loss = tf.reduce_mean(loss)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
    acc   = tf.reduce_mean(tf.cast(tf.equal(preds, batch['labels']), tf.float32))
    return loss, acc


@tf.function
def eval_step(model, batch):
    logits = model(batch, training=False)
    loss = tf.keras.losses.sparse_categorical_crossentropy(
        batch['labels'], logits, from_logits=True)
    loss = tf.reduce_mean(loss)
    preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
    acc   = tf.reduce_mean(tf.cast(tf.equal(preds, batch['labels']), tf.float32))
    return loss, acc


# ── Fine-tune one model ──────────────────────────────────────────────────────

def finetune_one_model(name, encoder, color, train_ds, val_ds, args, log_dir,
                       total_steps, warmup_steps):
    """Fine-tune a pretrained encoder on SST-2.

    Returns dict with accuracy, loss, etc.
    """
    print(f"\n{C.BOLD}{'═' * 65}{C.RESET}")
    print(f"  {color}{C.BOLD}▶  Fine-tuning: {name.upper()} on SST-2{C.RESET}")
    print(f"{C.BOLD}{'═' * 65}{C.RESET}")

    # Build classifier
    classifier = BertClassifier(
        encoder, num_classes=2, dropout_rate=args.dropout_rate,
        name=f'{name}_cls')

    # Warm up classifier head
    dummy = {
        'input_ids':      tf.zeros([2, args.finetune_seq_len], dtype=tf.int32),
        'attention_mask':  tf.ones([2, args.finetune_seq_len], dtype=tf.int32),
        'labels':         tf.zeros([2], dtype=tf.int32),
    }
    _ = classifier(dummy, training=False)

    enc_params = sum(int(tf.size(v).numpy())
                     for v in encoder.trainable_variables)
    cls_params = sum(int(tf.size(v).numpy())
                     for v in classifier.trainable_variables)
    print(f"  Encoder params  : {enc_params:,}")
    print(f"  Total params    : {cls_params:,}")
    print(f"  Classifier head : {cls_params - enc_params:,}")

    # Logger
    log_path = os.path.join(log_dir, f'finetune-{name}.log')
    csv_path = os.path.join(log_dir, f'finetune-{name}.csv')
    logger = setup_logger(f'ft_{name}', log_path)
    logger.info(f"Fine-tune started | model={name} | epochs={args.epochs} | "
                f"batch={args.batch_size} | lr={args.learning_rate}")

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
        _lo = classifier(_wb, training=True)
        _ls = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
            _wb['labels'], _lo, from_logits=True))
    _g = _t.gradient(_ls, classifier.trainable_variables)
    optimizer.apply_gradients(zip(_g, classifier.trainable_variables))
    optimizer.iterations.assign(0)
    del _wb, _lo, _ls, _g, _t

    best_val_acc  = 0.0
    best_epoch    = 0
    global_step   = 0

    for epoch in range(1, args.epochs + 1):
        # ── Training ─────────────────────────────────────────────────────
        tr_losses, tr_accs = [], []
        t0 = time.time()

        for batch in train_ds:
            loss, acc = train_step(classifier, optimizer, batch)
            tr_losses.append(float(loss))
            tr_accs.append(float(acc))
            global_step += 1

        elapsed = time.time() - t0
        n_steps = max(len(tr_losses), 1)
        sps     = n_steps / max(elapsed, 1e-6)

        # ── Validation ───────────────────────────────────────────────────
        va_losses, va_accs = [], []
        for batch in val_ds:
            vl, va = eval_step(classifier, batch)
            va_losses.append(float(vl))
            va_accs.append(float(va))

        t_loss = sum(tr_losses) / n_steps
        t_acc  = sum(tr_accs)  / n_steps
        v_loss = sum(va_losses) / max(len(va_losses), 1)
        v_acc  = sum(va_accs)  / max(len(va_accs),  1)

        lr_now = float(lr_schedule(optimizer.iterations))

        # Log
        logger.info(
            f"Epoch {epoch}/{args.epochs} | LR: {lr_now:.3e} | "
            f"Train loss={t_loss:.4f} acc={t_acc:.4f} | "
            f"Val loss={v_loss:.4f} acc={v_acc:.4f} | "
            f"{sps:.1f} step/s")

        # CSV
        write_header = not os.path.exists(csv_path)
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=[
                'epoch', 'global_step', 'lr', 'train_loss', 'train_acc',
                'val_loss', 'val_acc', 'steps_per_second', 'elapsed_s'])
            if write_header:
                w.writeheader()
            w.writerow({
                'epoch': epoch, 'global_step': global_step,
                'lr': round(lr_now, 8),
                'train_loss': round(t_loss, 6), 'train_acc': round(t_acc, 6),
                'val_loss': round(v_loss, 6), 'val_acc': round(v_acc, 6),
                'steps_per_second': round(sps, 4),
                'elapsed_s': round(elapsed, 2),
            })

        # Console
        print(f"  {color}[{name}]{C.RESET}"
              f"  Epoch {epoch:2d}/{args.epochs}"
              f"  | LR: {lr_now:.2e}"
              f"  | Train: loss={t_loss:.4f} acc={t_acc:.4f}"
              f"  | Val: loss={v_loss:.4f} acc={v_acc*100:.2f}%"
              f"  | {sps:.1f} step/s")

        # Best checkpoint
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_epoch   = epoch
            classifier.save_weights(
                os.path.join(log_dir, f'best_classifier.weights.h5'))
            print(f"    {C.GREEN}↑ New best val_acc={v_acc*100:.2f}% "
                  f"(epoch {epoch}){C.RESET}")

    logger.info(f"Fine-tune complete! Best val_acc={best_val_acc*100:.2f}% "
                f"at epoch {best_epoch}")

    return {
        'model': name,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'final_val_loss': v_loss,
        'final_val_acc': v_acc,
        'final_train_loss': t_loss,
        'final_train_acc': t_acc,
        'steps_per_second': sps,
        'global_step': global_step,
    }


# ── Final comparison + Go/No-Go ──────────────────────────────────────────────

def print_comparison(results, acc_threshold=1.5):
    """Print SST-2 comparison and Go/No-Go verdict."""
    print(f"\n{C.BOLD}{'═' * 70}{C.RESET}")
    print(f"{C.BOLD}  SST-2 Fine-tuning Results{C.RESET}")
    print(f"{C.BOLD}{'═' * 70}{C.RESET}")

    header = (f"  {'Model':<18} {'Best Acc':>10} {'Best Ep':>8} "
              f"{'Val Loss':>9} {'Steps/s':>8}")
    print(header)
    print(f"  {'─'*18} {'─'*10} {'─'*8} {'─'*9} {'─'*8}")

    color_map = {'vanilla': C.BLUE, 'tokendrop': C.YELLOW,
                 'progressive': C.GREEN}
    for r in results:
        c = color_map.get(r['model'], '')
        print(f"  {c}{r['model']:<18}{C.RESET}"
              f" {r['best_val_acc']*100:>9.2f}%"
              f" {r['best_epoch']:>8}"
              f" {r['final_val_loss']:>9.4f}"
              f" {r.get('steps_per_second', 0):>8.1f}")

    # Go/No-Go: Progressive acc ≥ vanilla - threshold
    vanilla_r = next((r for r in results if r['model'] == 'vanilla'), None)
    prog_r    = next((r for r in results if r['model'] == 'progressive'), None)

    if vanilla_r and prog_r:
        van_acc  = vanilla_r['best_val_acc'] * 100
        prog_acc = prog_r['best_val_acc'] * 100
        delta    = prog_acc - van_acc
        ok       = delta >= -acc_threshold

        print(f"\n  {C.BOLD}Go/No-Go SST-2 Evaluation:{C.RESET}")
        sym = f"{C.GREEN}✓{C.RESET}" if ok else f"{C.RED}✗{C.RESET}"
        print(f"  {sym}  Progressive acc ≥ vanilla − {acc_threshold:.1f}%")
        print(f"      vanilla={van_acc:.2f}%  progressive={prog_acc:.2f}%  "
              f"Δ={delta:+.2f}%  threshold=−{acc_threshold:.1f}%")

        if ok:
            print(f"\n  {C.GREEN}{C.BOLD}✅ GO — SST-2 criterion met{C.RESET}")
        else:
            print(f"\n  {C.RED}{C.BOLD}❌ NO-GO — Progressive accuracy "
                  f"below threshold{C.RESET}")

    print(f"{C.BOLD}{'═' * 70}{C.RESET}")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='SST-2 Fine-tuning from Pretrained MLM Checkpoints',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Paths
    p.add_argument('--checkpoint_dir',
        default='/data/umityilmaz/token_drop_l2/checkpoints/train_base512',
        help='Directory containing vanilla/, tokendrop/, progressive/ subdirs')
    p.add_argument('--output_dir',
        default='/data/umityilmaz/token_drop_l2/checkpoints/finetune_sst2',
        help='Output directory for fine-tuning results')
    p.add_argument('--cache_dir',
        default='/data/umityilmaz/token_drop_l2/data/hf_cache',
        help='HuggingFace cache directory')

    # Fine-tuning
    p.add_argument('--epochs',        type=int,   default=5)
    p.add_argument('--batch_size',    type=int,   default=32)
    p.add_argument('--learning_rate', type=float, default=2e-5,
                   help='Peak LR (standard: 2e-5 for BERT fine-tuning)')
    p.add_argument('--warmup_ratio',  type=float, default=0.1)
    p.add_argument('--weight_decay',  type=float, default=0.01)
    p.add_argument('--dropout_rate',  type=float, default=0.1)

    # Architecture (MUST match pretraining config)
    p.add_argument('--hidden_size',       type=int, default=768)
    p.add_argument('--num_layers',        type=int, default=12)
    p.add_argument('--num_heads',         type=int, default=12)
    p.add_argument('--intermediate_size', type=int, default=3072)
    p.add_argument('--pretrain_seq_len',  type=int, default=512,
                   help='Max seq len used during pretraining (for positional embeddings)')
    p.add_argument('--finetune_seq_len',  type=int, default=128,
                   help='Sequence length for SST-2 tokenization')

    # Control
    p.add_argument('--models', nargs='+',
                   default=['vanilla', 'tokendrop', 'progressive'],
                   choices=['vanilla', 'tokendrop', 'progressive'])
    p.add_argument('--acc_threshold', type=float, default=1.5,
                   help='Max allowed accuracy drop vs vanilla (percentage points)')
    p.add_argument('--seed', type=int, default=42)

    return p.parse_args()


def main():
    args = parse_args()

    # GPU memory growth
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Set seed for reproducibility
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Print config ──────────────────────────────────────────────────────────
    print(f"\n{C.CYAN}{C.BOLD}{'─'*65}")
    print(f"  SST-2 Fine-tuning — Go/No-Go #3 Downstream Evaluation")
    print(f"{'─'*65}{C.RESET}")
    print(f"  Checkpoint  : {args.checkpoint_dir}")
    print(f"  Output      : {args.output_dir}")
    print(f"  Models      : {', '.join(args.models)}")
    print(f"  Architecture: hidden={args.hidden_size}, layers={args.num_layers}, "
          f"heads={args.num_heads}")
    print(f"  Pretrain seq: {args.pretrain_seq_len}  |  Fine-tune seq: "
          f"{args.finetune_seq_len}")
    ft_seq = args.finetune_seq_len
    print(f"  Token budgets (fine-tune): "
          f"tokendrop k={ft_seq//2}, "
          f"progressive k1={int(ft_seq*0.75)}/k2={ft_seq//2}/k3={ft_seq//4}")
    print(f"  Training    : epochs={args.epochs}, batch={args.batch_size}, "
          f"lr={args.learning_rate}, wd={args.weight_decay}")

    # ── Load SST-2 data ──────────────────────────────────────────────────────
    train_data, val_data = load_sst2_data(
        max_seq_len=args.finetune_seq_len, cache_dir=args.cache_dir)

    train_ds = make_tf_dataset(train_data, args.batch_size, shuffle=True,
                               seed=args.seed)
    val_ds   = make_tf_dataset(val_data, args.batch_size, shuffle=False)

    # LR schedule
    n_train         = len(train_data['input_ids'])
    steps_per_epoch = n_train // args.batch_size
    total_steps     = args.epochs * steps_per_epoch
    warmup_steps    = int(total_steps * args.warmup_ratio)
    print(f"  LR schedule : total_steps={total_steps:,}  "
          f"warmup={warmup_steps:,}  ({args.warmup_ratio:.0%})")

    # ── Fine-tune each model ─────────────────────────────────────────────────
    color_map = {'vanilla': C.BLUE, 'tokendrop': C.YELLOW,
                 'progressive': C.GREEN}
    all_results = []

    for model_name in args.models:
        color   = color_map.get(model_name, '')
        log_dir = os.path.join(args.output_dir, model_name)
        os.makedirs(log_dir, exist_ok=True)

        # Build encoder + load pretrained weights
        encoder = build_and_load_encoder(model_name, args)

        # Fine-tune
        result = finetune_one_model(
            model_name, encoder, color, train_ds, val_ds, args, log_dir,
            total_steps=total_steps, warmup_steps=warmup_steps)
        all_results.append(result)

        # Clear model to free GPU memory before next model
        tf.keras.backend.clear_session()

    # ── Results ──────────────────────────────────────────────────────────────
    print_comparison(all_results, acc_threshold=args.acc_threshold)

    # Save results JSON
    payload = {
        'timestamp': datetime.datetime.now().isoformat(),
        'args': vars(args),
        'results': {r['model']: r for r in all_results},
    }
    json_path = os.path.join(args.output_dir, 'sst2_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    print(f"\n  {C.CYAN}Results saved:{C.RESET} {json_path}")

    # Save results CSV
    csv_path = os.path.join(args.output_dir, 'sst2_results.csv')
    fields = ['model', 'best_val_acc', 'best_epoch', 'final_val_loss',
              'final_train_loss', 'steps_per_second']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in all_results:
            w.writerow(r)
    print(f"  {C.CYAN}CSV summary:{C.RESET}  {csv_path}")


if __name__ == '__main__':
    main()
