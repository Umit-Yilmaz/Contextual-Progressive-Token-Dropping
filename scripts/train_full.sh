#!/usr/bin/env bash
# =============================================================================
# train_full.sh  —  Aşama 3: Tam Eğitim (1,000,000 adım)
#
# SADECE Aşama 2'den GO aldıktan sonra çalıştırın.
#
# Seq-len=512, global batch=256 (16 örnek × grad_accum=16)
# Tek GPU → gradient accumulation ile efektif batch büyütülür.
# Tahmini süre: ~4 gün (A100 80GB'da)
#
# Kullanım:
#   bash scripts/train_full.sh \
#     --train_data /path/to/tfrecords/train/*.tfrecord \
#     --eval_data  /path/to/tfrecords/eval/*.tfrecord
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$REPO_ROOT/tfenv/Scripts/python}"
TRAIN_SCRIPT="$REPO_ROOT/train.py"

TRAIN_DATA="${TRAIN_DATA:-/path-to-packed-data/train/*.tfrecord}"
EVAL_DATA="${EVAL_DATA:-/path-to-packed-data/eval/*.tfrecord}"
CKPT_DIR="${CKPT_DIR:-$REPO_ROOT/checkpoints}"
ONLY="${ONLY:-both}"   # "baseline", "progressive" veya "both"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --train_data) TRAIN_DATA="$2"; shift 2 ;;
    --eval_data)  EVAL_DATA="$2";  shift 2 ;;
    --ckpt_dir)   CKPT_DIR="$2";   shift 2 ;;
    --only)       ONLY="$2";       shift 2 ;;
    *) echo "[ERROR] Bilinmeyen argüman: $1" >&2; exit 1 ;;
  esac
done

# ─── Ortak override (seq=512, 1M adım) ───────────────────────────────────────
DATA_OVERRIDE="task.train_data.seq_length=512,\
task.train_data.max_predictions_per_seq=76,\
task.train_data.global_batch_size=16,\
task.validation_data.seq_length=512,\
task.validation_data.max_predictions_per_seq=76,\
task.validation_data.global_batch_size=16,\
task.train_data.input_path='$TRAIN_DATA',\
task.validation_data.input_path='$EVAL_DATA'"

STEP_OVERRIDE="trainer.train_steps=1000000,\
trainer.optimizer_config.warmup.polynomial.warmup_steps=10000,\
trainer.optimizer_config.learning_rate.polynomial.decay_steps=1000000,\
trainer.validation_interval=5000,\
trainer.checkpoint_interval=20000,\
trainer.summary_interval=1000"

# ─── Baseline tam eğitim ──────────────────────────────────────────────────────
run_baseline_full() {
  echo ""
  echo "━━━ BASELINE TAM EĞİTİM (1M adım, seq=512) ━━━━━━━━━━━━━━━"
  echo "  Başlangıç: $(date)"
  $PYTHON "$TRAIN_SCRIPT" \
    --experiment=token_drop_bert/pretraining \
    --config_file="$REPO_ROOT/wiki_books_pretrain_sequence_pack.yaml" \
    --config_file="$REPO_ROOT/bert_en_uncased_base_token_drop.yaml" \
    --params_override="$DATA_OVERRIDE,$STEP_OVERRIDE" \
    --model_dir="$CKPT_DIR/full_baseline" \
    --mode=train_and_eval
  echo "  Bitiş: $(date)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# ─── Progressive tam eğitim ───────────────────────────────────────────────────
run_progressive_full() {
  echo ""
  echo "━━━ PROGRESSIVE TAM EĞİTİM (1M adım, seq=512) ━━━━━━━━━━━━"
  echo "  Başlangıç: $(date)"
  $PYTHON "$TRAIN_SCRIPT" \
    --experiment=progressive_drop_bert/pretraining \
    --config_file="$REPO_ROOT/wiki_books_pretrain_sequence_pack.yaml" \
    --config_file="$REPO_ROOT/experiments/progressive_contextual_dropping/bert_progressive_drop.yaml" \
    --params_override="$DATA_OVERRIDE,$STEP_OVERRIDE" \
    --model_dir="$CKPT_DIR/full_progressive" \
    --mode=train_and_eval
  echo "  Bitiş: $(date)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

echo "============================================================"
echo " TAM EĞİTİM (Aşama 3)"
echo " Sadece: $ONLY"
echo " Tahmini süre: ~4 gün"
echo " Başlangıç: $(date)"
echo ""
echo " NOT: Bu koşu sadece Aşama 2 Go/No-Go kararından sonra"
echo " başlatılmalıdır. İptal etmek için Ctrl+C."
echo "============================================================"

# 10 saniye bekleme — iptal fırsatı
for i in $(seq 10 -1 1); do
  echo -ne "  $i saniye sonra başlıyor... (Ctrl+C ile iptal)\r"
  sleep 1
done
echo ""

case "$ONLY" in
  baseline)    run_baseline_full ;;
  progressive) run_progressive_full ;;
  both)        run_baseline_full; run_progressive_full ;;
  *) echo "[ERROR] --only değeri baseline, progressive veya both olmalı." >&2; exit 1 ;;
esac

echo ""
echo "============================================================"
echo " Tam eğitim tamamlandı: $(date)"
echo ""
echo " Checkpoint konumları:"
echo "   Baseline:    $CKPT_DIR/full_baseline/"
echo "   Progressive: $CKPT_DIR/full_progressive/"
echo ""
echo " Sonraki adım — GLUE değerlendirmesi:"
echo "   bash scripts/run_glue.sh \\"
echo "     --baseline_ckpt=$CKPT_DIR/full_baseline/ \\"
echo "     --progressive_ckpt=$CKPT_DIR/full_progressive/"
echo "============================================================"
