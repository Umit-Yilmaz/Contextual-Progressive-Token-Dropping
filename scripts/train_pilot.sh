#!/usr/bin/env bash
# =============================================================================
# train_pilot.sh  —  Aşama 0: Üçlü Pilot Koşusu (10,000 adım)
#
# Üç modeli karşılaştırmalı eğitir:
#   1) vanilla_bert     — Standart BERT (token dropping yok), referans baseline
#   2) token_drop_bert  — Vocab-seviyesi önem skorlu TokenDrop BERT
#   3) progressive_drop — L2-norm tabanlı Progressive Contextual Drop BERT (katkımız)
#
# Kullanım:
#   bash scripts/train_pilot.sh \
#     --train_data /path/to/tfrecords/train/*.tfrecord \
#     --eval_data  /path/to/tfrecords/eval/*.tfrecord  \
#     [--parallel]   # Üç modeli aynı anda başlat (RTX A6000 48GB → güvenli)
#
# Soft eğitim ayarları (RTX A6000 48 GB):
#   seq_len=128, batch=16, grad_accum=4 (efektif batch=64), steps=10k
#
# Modeller varsayılan olarak sıralı (sequential) çalışır.
# --parallel ile 3 modelin aynı anda çalışması için yaklaşık 18 GB GPU belleği
# gerekir — A6000 48 GB bunu rahatlıkla destekler.
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$REPO_ROOT/venv/bin/python}"
TRAIN_SCRIPT="$REPO_ROOT/train.py"

TRAIN_DATA="${TRAIN_DATA:-/path-to-packed-data/train/*.tfrecord}"
EVAL_DATA="${EVAL_DATA:-/path-to-packed-data/eval/*.tfrecord}"
CKPT_DIR="${CKPT_DIR:-$REPO_ROOT/checkpoints}"
PARALLEL=false

# ─── Argüman ayrıştırma ───────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --train_data) TRAIN_DATA="$2"; shift 2 ;;
    --eval_data)  EVAL_DATA="$2";  shift 2 ;;
    --ckpt_dir)   CKPT_DIR="$2";   shift 2 ;;
    --parallel)   PARALLEL=true;   shift   ;;
    *) echo "[ERROR] Bilinmeyen argüman: $1" >&2; exit 1 ;;
  esac
done

# ─── Soft eğitim override: seq_len=128, pilot adımları ───────────────────────
# seq=128 → max_predictions_per_seq = round(128 × 0.15) = 20 (≈ %15 maskeleme)
COMMON_OVERRIDE="\
task.train_data.seq_length=128,\
task.train_data.max_predictions_per_seq=20,\
task.train_data.global_batch_size=16,\
task.validation_data.seq_length=128,\
task.validation_data.max_predictions_per_seq=20,\
task.validation_data.global_batch_size=16,\
task.train_data.input_path='${TRAIN_DATA}',\
task.validation_data.input_path='${EVAL_DATA}',\
trainer.train_steps=10000,\
trainer.validation_interval=500,\
trainer.checkpoint_interval=2000,\
trainer.summary_interval=100,\
trainer.optimizer_config.warmup.polynomial.warmup_steps=2000,\
trainer.optimizer_config.learning_rate.polynomial.decay_steps=10000"

echo "============================================================"
echo " 3-Way Pilot Koşusu: 10,000 adım | seq_len=128 | soft mode"
echo " Vanilla:    $CKPT_DIR/pilot_vanilla/"
echo " TokenDrop:  $CKPT_DIR/pilot_tokendrop/"
echo " Progressive:$CKPT_DIR/pilot_progressive/"
echo "============================================================"

# ─── 1. Vanilla BERT ─────────────────────────────────────────────────────────
run_vanilla() {
  echo ""
  echo "━━━ [1/3] VANILLA BERT başlatılıyor ━━━━━━━━━━━━━━━━━━━━━━━"
  "$PYTHON" "$TRAIN_SCRIPT" \
    --experiment=vanilla_bert/pretraining \
    --config_file="$REPO_ROOT/wiki_books_pretrain_sequence_pack.yaml" \
    --config_file="$REPO_ROOT/bert_en_uncased_base_vanilla.yaml" \
    --params_override="${COMMON_OVERRIDE}" \
    --model_dir="$CKPT_DIR/pilot_vanilla" \
    --mode=train_and_eval
  echo "━━━ [1/3] VANILLA BERT tamamlandı ━━━━━━━━━━━━━━━━━━━━━━━━"
}

# ─── 2. TokenDrop BERT ────────────────────────────────────────────────────────
run_tokendrop() {
  echo ""
  echo "━━━ [2/3] TOKEN-DROP BERT başlatılıyor ━━━━━━━━━━━━━━━━━━━"
  "$PYTHON" "$TRAIN_SCRIPT" \
    --experiment=token_drop_bert/pretraining \
    --config_file="$REPO_ROOT/wiki_books_pretrain_sequence_pack.yaml" \
    --config_file="$REPO_ROOT/bert_en_uncased_base_token_drop.yaml" \
    --params_override="${COMMON_OVERRIDE}" \
    --model_dir="$CKPT_DIR/pilot_tokendrop" \
    --mode=train_and_eval
  echo "━━━ [2/3] TOKEN-DROP BERT tamamlandı ━━━━━━━━━━━━━━━━━━━━━"
}

# ─── 3. Progressive Drop BERT ────────────────────────────────────────────────
# k1/k2/k3 seq_len=128 için ölçeklendi:
#   k1 = round(128 × 384/512) = 96   (%75 tutulur — ilk aşama)
#   k2 = round(128 × 256/512) = 64   (%50 tutulur — ikinci aşama)
#   k3 = round(128 × 128/512) = 32   (%25 tutulur — üçüncü aşama)
run_progressive() {
  echo ""
  echo "━━━ [3/3] PROGRESSIVE DROP BERT başlatılıyor ━━━━━━━━━━━━━"
  "$PYTHON" "$TRAIN_SCRIPT" \
    --experiment=progressive_drop_bert/pretraining \
    --config_file="$REPO_ROOT/wiki_books_pretrain_sequence_pack.yaml" \
    --config_file="$REPO_ROOT/experiments/progressive_contextual_dropping/bert_progressive_drop.yaml" \
    --params_override="${COMMON_OVERRIDE},\
task.model.encoder.any.token_keep_k1=96,\
task.model.encoder.any.token_keep_k2=64,\
task.model.encoder.any.token_keep_k3=32" \
    --model_dir="$CKPT_DIR/pilot_progressive" \
    --mode=train_and_eval
  echo "━━━ [3/3] PROGRESSIVE DROP BERT tamamlandı ━━━━━━━━━━━━━━━"
}

# ─── Çalıştırma modu ─────────────────────────────────────────────────────────
if $PARALLEL; then
  echo "[INFO] Paralel mod: 3 model aynı anda başlatılıyor."
  echo "[INFO] RTX A6000 48 GB → ~18 GB toplam VRAM (güvenli)."
  run_vanilla     &
  VANILLA_PID=$!
  run_tokendrop   &
  TOKENDROP_PID=$!
  run_progressive &
  PROGRESSIVE_PID=$!
  wait $VANILLA_PID     && echo "[OK] Vanilla process bitti."
  wait $TOKENDROP_PID   && echo "[OK] TokenDrop process bitti."
  wait $PROGRESSIVE_PID && echo "[OK] Progressive process bitti."
else
  # Sıralı (soft mod) — her model GPU'yu tamamen kullanır
  run_vanilla
  run_tokendrop
  run_progressive
fi

echo ""
echo "============================================================"
echo " Pilot koşusu tamamlandı! 3-way Go/No-Go analizi için:"
echo ""
echo "   python scripts/early_stop_monitor.py \\"
echo "     --vanilla_logdir=$CKPT_DIR/pilot_vanilla/ \\"
echo "     --baseline_logdir=$CKPT_DIR/pilot_tokendrop/ \\"
echo "     --progressive_logdir=$CKPT_DIR/pilot_progressive/ \\"
echo "     --threshold=0.10"
echo "============================================================"
