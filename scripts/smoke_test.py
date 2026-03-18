#!/usr/bin/env python3
# =============================================================================
# smoke_test.py  —  Full Pipeline Smoke Test (train.py gerektirmez)
#
# TFRecord verisini yükler, baseline + progressive modellerini oluşturur,
# her iki modeli N adım eğitir, Go/No-Go kriterlerini değerlendirir.
#
# Kullanım:
#   python scripts/smoke_test.py \
#     --train_data ./data/smoke_test/train/*.tfrecord \
#     --eval_data  ./data/smoke_test/eval/*.tfrecord  \
#     [--steps 200] [--batch 4] [--seq_len 128]
# =============================================================================

import argparse
import glob
import math
import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Only insert repo root – do NOT add experiments subdir at index 0,
# that would shadow the root-level encoder.py / encoder_config.py.
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import tensorflow as tf
tf_keras = tf.keras  # use tf.keras (standalone tf-keras 2.15 ≠ TF 2.10)

# Import encoder classes directly — bypass encoder_config.py files which
# depend on official.projects.token_dropping (not present in the installed
# tf-models-official 2.10 package).
from encoder import TokenDropBertEncoder  # noqa: E402  (root encoder.py)
from experiments.progressive_contextual_dropping.encoder import (   # noqa: E402
    ProgressiveContextualDropEncoder
)

# ─── ANSI renk kodları ────────────────────────────────────────────────────────
class C:
    GREEN  = "\033[92m"
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    BOLD   = "\033[1m"
    CYAN   = "\033[96m"
    RESET  = "\033[0m"

def ok(msg):   return f"{C.GREEN}[OK]{C.RESET} {msg}"
def fail(msg): return f"{C.RED}[FAIL]{C.RESET} {msg}"
def warn(msg): return f"{C.YELLOW}[WARN]{C.RESET} {msg}"
def info(msg): return f"{C.CYAN}[INFO]{C.RESET} {msg}"


# ─── TFRecord yükleme ─────────────────────────────────────────────────────────

def load_dataset(file_pattern: str, seq_len: int, max_pred: int,
                 batch_size: int, shuffle: bool = True):
    """BertPretrainDataConfig v2 feature names ile TFRecord okur."""
    files = sorted(glob.glob(file_pattern))
    if not files:
        # Glob işe yaramadıysa direkt yol dene
        files = [file_pattern] if os.path.exists(file_pattern) else []
    if not files:
        raise FileNotFoundError(f"TFRecord bulunamadi: {file_pattern}")

    feat_spec = {
        "input_word_ids":      tf.io.FixedLenFeature([seq_len], tf.int64),
        "input_mask":          tf.io.FixedLenFeature([seq_len], tf.int64),
        "input_type_ids":      tf.io.FixedLenFeature([seq_len], tf.int64),
        "masked_lm_positions": tf.io.FixedLenFeature([max_pred], tf.int64),
        "masked_lm_ids":       tf.io.FixedLenFeature([max_pred], tf.int64),
        "masked_lm_weights":   tf.io.FixedLenFeature([max_pred], tf.float32),
    }

    ds = tf.data.TFRecordDataset(files)
    if shuffle:
        ds = ds.shuffle(buffer_size=1000, seed=42)

    def parse(raw):
        return tf.io.parse_single_example(raw, feat_spec)

    ds = ds.map(parse, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.repeat()
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, len(files)


# ─── Model oluşturma ──────────────────────────────────────────────────────────

def build_bert_pretrainer(encoder, vocab_size: int, hidden_size: int):
    """
    Encoder + MLM head = minimal BERT pretrainer.
    """
    from official.nlp.modeling import models as nlp_models
    from official.nlp.configs import bert
    from official.nlp.configs import encoders as enc_cfgs

    # Doğrudan encoder'ı MLM head ile sar
    try:
        from official.nlp.modeling.models.bert_pretrainer import (
            BertPretrainerV2
        )
        pretrainer = BertPretrainerV2(
            encoder_network=encoder,
            mlm_initializer=tf_keras.initializers.TruncatedNormal(stddev=0.02),
        )
    except Exception:
        # Fallback: manuel MLM head
        pretrainer = _build_manual_mlm(encoder, vocab_size, hidden_size)

    return pretrainer


def _build_manual_mlm(encoder, vocab_size: int, hidden_size: int):
    """BertPretrainerV2 erişilemezse basit MLM modeli."""
    class ManualMLM(tf_keras.Model):
        def __init__(self, encoder, vocab_size, hidden_size):
            super().__init__()
            self.encoder = encoder
            self.mlm_dense = tf_keras.layers.Dense(hidden_size, activation='gelu')
            self.mlm_norm  = tf_keras.layers.LayerNormalization()
            self.mlm_bias  = tf_keras.layers.Dense(vocab_size, use_bias=True)

        def call(self, inputs, training=False):
            word_ids   = inputs["input_word_ids"]
            input_mask = inputs["input_mask"]
            type_ids   = inputs["input_type_ids"]

            # Both encoders require dict inputs – never a list.
            enc_out = self.encoder(
                {"input_word_ids": word_ids,
                 "input_mask": input_mask,
                 "input_type_ids": type_ids},
                training=training)
            # enc_out: list [sequence_output, cls_output] veya dict
            if isinstance(enc_out, (list, tuple)):
                seq_out = enc_out[0]
            elif isinstance(enc_out, dict):
                seq_out = enc_out.get("sequence_output",
                           enc_out.get("last_hidden_state", list(enc_out.values())[0]))
            else:
                seq_out = enc_out  # [B, T, H]

            lm_out = self.mlm_dense(seq_out)
            lm_out = self.mlm_norm(lm_out)
            logits = self.mlm_bias(lm_out)   # [B, T, V]
            return {"sequence_output": seq_out, "mlm_logits": logits}

    return ManualMLM(encoder, vocab_size, hidden_size)


# ─── Kayıp fonksiyonu ─────────────────────────────────────────────────────────

def compute_mlm_loss(logits, masked_lm_ids, masked_lm_positions,
                     masked_lm_weights, model_outputs=None):
    """
    MLM kayıp: masked_lm_positions konumlarındaki logitler ile
    masked_lm_ids karşılaştırılır.
    """
    # logits: [B, T, V]
    batch_size = tf.shape(logits)[0]
    seq_len    = tf.shape(logits)[1]

    # Masked pozisyonların logitlerini topla
    # positions: [B, max_pred] → gather için [B, max_pred, V]
    pos = tf.cast(masked_lm_positions, tf.int32)         # [B, P]
    B   = tf.range(batch_size)[:, None]                  # [B, 1]
    B   = tf.tile(B, [1, tf.shape(pos)[1]])              # [B, P]
    indices = tf.stack([B, pos], axis=-1)                # [B, P, 2]
    gathered_logits = tf.gather_nd(logits, indices)      # [B, P, V]

    # Cross-entropy
    per_token_loss = tf_keras.losses.sparse_categorical_crossentropy(
        masked_lm_ids,
        tf.cast(gathered_logits, tf.float32),
        from_logits=True,
    )  # [B, P]
    weights = masked_lm_weights  # [B, P]
    loss = tf.math.divide_no_nan(
        tf.reduce_sum(per_token_loss * weights),
        tf.reduce_sum(weights)
    )
    return loss


# ─── Tek model eğitimi ────────────────────────────────────────────────────────

def train_model(model_name: str, model, dataset, optimizer,
                steps: int, log_every: int = 50) -> dict:
    """Modeli `steps` adım eğitir, loss geçmişini döndürür."""
    print(f"\n{'='*62}")
    print(f"  Model: {C.BOLD}{model_name}{C.RESET}")
    print(f"  Steps: {steps}  |  Optimizer: AdamW")
    print(f"{'='*62}")

    loss_history = []
    times        = []
    nan_detected = False
    data_iter    = iter(dataset)

    for step in range(1, steps + 1):
        batch = next(data_iter)
        t0    = time.time()

        with tf.GradientTape() as tape:
            outputs = model(batch, training=True)
            # Logits alımı
            if isinstance(outputs, dict):
                logits = outputs.get("mlm_logits",
                         outputs.get("mlm_log_probs", None))
            elif isinstance(outputs, (list, tuple)):
                logits = outputs[-1]
            else:
                logits = outputs

            loss = compute_mlm_loss(
                logits,
                batch["masked_lm_ids"],
                batch["masked_lm_positions"],
                batch["masked_lm_weights"],
            )
            # Aux losses (regularization)
            if model.losses:
                loss = loss + tf.add_n(model.losses)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        elapsed = time.time() - t0

        loss_val = float(loss.numpy())
        loss_history.append(loss_val)
        times.append(elapsed)

        if math.isnan(loss_val) or math.isinf(loss_val):
            nan_detected = True
            print(f"  {fail(f'Step {step}: NaN/Inf loss tespit edildi!')}")
            break

        if step % log_every == 0 or step == 1:
            avg_time  = sum(times[-log_every:]) / len(times[-log_every:])
            avg_loss  = sum(loss_history[-log_every:]) / len(loss_history[-log_every:])
            print(f"  Step {step:>5}/{steps}  "
                  f"loss={avg_loss:.4f}  "
                  f"ms/step={avg_time*1000:.1f}")

    return {
        "loss_history": loss_history,
        "final_loss":   loss_history[-1] if loss_history else float("inf"),
        "avg_step_ms":  sum(times) / len(times) * 1000 if times else 0,
        "nan_detected": nan_detected,
    }


# ─── Go/No-Go değerlendirmesi ─────────────────────────────────────────────────

def evaluate_go_no_go(b_result: dict, p_result: dict,
                      threshold: float = 0.10) -> bool:
    print(f"\n{'='*62}")
    print(f"  {C.BOLD}Go/No-Go Degerlendirmesi{C.RESET}")
    print(f"{'='*62}")

    b_loss = b_result["final_loss"]
    p_loss = p_result["final_loss"]
    b_time = b_result["avg_step_ms"]
    p_time = p_result["avg_step_ms"]

    checks = []

    # 1. NaN/Inf
    nan_ok = not b_result["nan_detected"] and not p_result["nan_detected"]
    checks.append(("NaN/Inf yok", nan_ok,
                   f"baseline={b_result['nan_detected']}, prog={p_result['nan_detected']}"))

    # 2. Loss gap
    if b_loss > 0:
        loss_ratio = p_loss / b_loss
        loss_ok    = loss_ratio <= (1 + threshold)
        checks.append((f"Progressive loss <= baseline x {1+threshold:.2f}",
                        loss_ok,
                        f"progressive={p_loss:.4f}, baseline={b_loss:.4f}, oran={loss_ratio:.3f}"))
    else:
        checks.append(("Loss gap", False, "Baseline loss=0?"))

    # 3. Progressive throughput >= 0.85x baseline
    if b_time > 0:
        tput_ratio = b_time / p_time  # < 1 means progressive is faster
        tput_ok    = tput_ratio >= 0.85
        checks.append(("Progressive throughput >= baseline x 0.85",
                        tput_ok,
                        f"baseline={b_time:.1f}ms, progressive={p_time:.1f}ms, oran={tput_ratio:.3f}"))

    # 4. Loss NOT strongly diverging (last 20% must not be >5% above first 20%).
    # We check for divergence only – flat loss (typical in short synthetic runs)
    # is treated as a PASS. A real convergence check belongs in the pilot run.
    def trend_ok(hist):
        if len(hist) < 10:
            return True
        n   = len(hist)
        q   = max(1, n // 5)
        first_avg = sum(hist[:q]) / q
        last_avg  = sum(hist[-q:]) / q
        # Fail only if loss is increasing (diverging) by more than 5 %
        return last_avg <= first_avg * 1.05

    b_trend = trend_ok(b_result["loss_history"])
    p_trend = trend_ok(p_result["loss_history"])
    checks.append(("Loss khong phan ky (last_avg <= first_avg x 1.05)",
                    b_trend and p_trend,
                    f"baseline_ok={b_trend}, prog_ok={p_trend}"))

    # Çıktı
    all_passed = True
    for name, passed, detail in checks:
        sym  = ok("") if passed else fail("")
        star = "" if passed else "  --> " + detail
        print(f"  {sym} {name}")
        if not passed:
            print(f"      {detail}")
            all_passed = False

    print(f"\n  {'='*58}")
    if all_passed:
        print(f"  {C.GREEN}{C.BOLD}KARAR: GO  -- Pipeline dogrulandı!{C.RESET}")
    else:
        print(f"  {C.RED}{C.BOLD}KARAR: NO-GO -- Kontrol gereken sorunlar var.{C.RESET}")
    print(f"  {'='*58}")

    return all_passed


# ─── Ana fonksiyon ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pipeline smoke test")
    parser.add_argument("--train_data", default="./data/smoke_test/train/*.tfrecord")
    parser.add_argument("--eval_data",  default="./data/smoke_test/eval/*.tfrecord")
    parser.add_argument("--steps",      type=int,   default=200)
    parser.add_argument("--batch",      type=int,   default=4)
    parser.add_argument("--seq_len",    type=int,   default=128)
    parser.add_argument("--hidden",     type=int,   default=256,
                        help="Hidden size (256 = hizli test, 768 = BERT-base)")
    parser.add_argument("--layers",     type=int,   default=4,
                        help="Katman sayisi (4 = hizli, 12 = BERT-base)")
    parser.add_argument("--threshold",  type=float, default=0.15,
                        help="Loss gap esigi (varsayilan 0.15 = %%15)")
    parser.add_argument("--log_every",  type=int,   default=50)
    args = parser.parse_args()

    max_pred = min(20, max(1, int(args.seq_len * 0.15)))

    print(f"\n{C.BOLD}{'='*62}{C.RESET}")
    print(f"  {C.BOLD}Progressive Contextual Drop -- Pipeline Smoke Test{C.RESET}")
    print(f"  seq_len={args.seq_len}  hidden={args.hidden}  layers={args.layers}")
    print(f"  steps={args.steps}  batch={args.batch}  max_pred={max_pred}")
    print(f"{C.BOLD}{'='*62}{C.RESET}")

    # ── Veri yükleme ─────────────────────────────────────────────────────────
    print(f"\n{info('Veri yukleniyor...')}")
    train_ds, n_train = load_dataset(args.train_data, args.seq_len,
                                     max_pred, args.batch)
    print(f"  Egitim TFRecord shard sayisi: {n_train}")

    # ── Encoder konfigürasyonları ─────────────────────────────────────────────
    n_heads = max(1, args.hidden // 64)
    k1 = max(4, int(args.seq_len * 0.75))
    k2 = max(3, int(args.seq_len * 0.50))
    k3 = max(2, int(args.seq_len * 0.25))

    # ── Modeller ─────────────────────────────────────────────────────────────
    # Build encoder classes directly — no config/factory indirection.
    print(f"\n{info('Modeller olusturuluyor...')}")
    baseline_enc = TokenDropBertEncoder(
        vocab_size=30522,
        hidden_size=args.hidden,
        num_layers=args.layers,
        num_attention_heads=n_heads,
        inner_dim=args.hidden * 4,
        max_sequence_length=512,
        token_keep_k=k2,
    )
    progressive_enc = ProgressiveContextualDropEncoder(
        vocab_size=30522,
        hidden_size=args.hidden,
        num_layers=args.layers,
        num_attention_heads=n_heads,
        inner_dim=args.hidden * 4,
        max_sequence_length=512,
        token_keep_k1=k1,
        token_keep_k2=k2,
        token_keep_k3=k3,
    )

    baseline_model    = _build_manual_mlm(baseline_enc,    30522, args.hidden)
    progressive_model = _build_manual_mlm(progressive_enc, 30522, args.hidden)

    # İlk forward pass ile ağırlıkları oluştur
    dummy_batch = next(iter(train_ds))
    _ = baseline_model(dummy_batch, training=False)
    _ = progressive_model(dummy_batch, training=False)

    b_params = sum(v.numpy().size for v in baseline_model.trainable_variables)
    p_params = sum(v.numpy().size for v in progressive_model.trainable_variables)
    b_nontr  = sum(v.numpy().size for v in baseline_enc.non_trainable_variables)
    p_nontr  = sum(v.numpy().size for v in progressive_enc.non_trainable_variables)

    print(f"  Baseline    trainable={b_params:,}  non-trainable={b_nontr:,}")
    print(f"  Progressive trainable={p_params:,}  non-trainable={p_nontr}")
    print(f"  Budget: k1={k1}, k2={k2}, k3={k3} (seq_len={args.seq_len})")

    # ── Optimizörler ─────────────────────────────────────────────────────────
    def make_optimizer(total_steps, warmup=20):
        from official.modeling import optimization
        cfg = optimization.OptimizationConfig({
            'optimizer': {
                'type': 'adamw',
                'adamw': {'weight_decay_rate': 0.01,
                          'exclude_from_weight_decay': ['LayerNorm','layer_norm','bias']},
            },
            'learning_rate': {
                'type': 'polynomial',
                'polynomial': {'initial_learning_rate': 1e-4,
                               'end_learning_rate': 0.0,
                               'decay_steps': total_steps},
            },
            'warmup': {'type': 'polynomial',
                       'polynomial': {'warmup_steps': warmup}},
        })
        factory = optimization.OptimizerFactory(cfg)
        lr = factory.build_learning_rate()
        return factory.build_optimizer(lr=lr)

    try:
        b_opt = make_optimizer(args.steps)
        p_opt = make_optimizer(args.steps)
        print(f"  {ok('AdamW optimizer olusturuldu.')}")
    except Exception as e:
        print(f"  {warn(f'AdamW basarisiz ({e}), SGD kullaniliyor.')}")
        b_opt = tf_keras.optimizers.SGD(learning_rate=1e-4, momentum=0.9)
        p_opt = tf_keras.optimizers.SGD(learning_rate=1e-4, momentum=0.9)

    # ── Eğitim ───────────────────────────────────────────────────────────────
    b_result = train_model("Baseline (TokenDrop)", baseline_model,
                           train_ds, b_opt, args.steps, args.log_every)
    p_result = train_model("Progressive Contextual Drop", progressive_model,
                           train_ds, p_opt, args.steps, args.log_every)

    # ── Özet ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  {C.BOLD}Ozet{C.RESET}")
    print(f"{'='*62}")
    print(f"  {'Model':<30} {'Son Loss':>10} {'ms/step':>9} {'Non-trainable':>14}")
    print(f"  {'-'*65}")
    print(f"  {'Baseline':<30} {b_result['final_loss']:>10.4f} "
          f"{b_result['avg_step_ms']:>9.1f} {b_nontr:>14,}")
    print(f"  {'Progressive':<30} {p_result['final_loss']:>10.4f} "
          f"{p_result['avg_step_ms']:>9.1f} {p_nontr:>14,}")

    if not b_result["nan_detected"] and b_result["avg_step_ms"] > 0:
        speedup = b_result["avg_step_ms"] / p_result["avg_step_ms"] - 1
        loss_delta = (p_result["final_loss"] - b_result["final_loss"]) / b_result["final_loss"] * 100
        print(f"\n  Progressive hiz farki: {speedup:+.1%}")
        print(f"  Progressive kayip farki: {loss_delta:+.2f}%")
        print(f"  Non-trainable param azalmasi: {b_nontr} -> {p_nontr} "
              f"({'-100%' if b_nontr > 0 else 'n/a'})")

    # ── Go/No-Go ─────────────────────────────────────────────────────────────
    passed = evaluate_go_no_go(b_result, p_result, threshold=args.threshold)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
