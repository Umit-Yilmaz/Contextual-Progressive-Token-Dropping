#!/usr/bin/env bash
# =============================================================================
# train_short.sh  —  Aşama 1 & 2: Kısa ve Orta Koşular
#
# Aşama 1: 50,000 adım   (~4 saat,  seq_len=128)
# Aşama 2: 200,000 adım  (~16 saat, seq_len=128)
#
# Her iki aşamayı sırayla çalıştırır; aşamalar arasında Go/No-Go kararı beklenir.
#
# Kullanım:
#   bash scripts/train_short.sh \
#     --train_data /path/to/train/*.tfrecord \
#     --eval_data  /path/to/eval/*.tfrecord  \
#     --phase 1    # sadece 50k adım (Aşama 1)
#     --phase 2    # 50k'dan devam ederek 200k'a (Aşama 2)
#     --phase all  # Her ikisini birden (varsayılan)
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$REPO_ROOT/tfenv/Scripts/python}"
TRAIN_SCRIPT="$REPO_ROOT/train.py"

TRAIN_DATA="${TRAIN_DATA:-/path-to-packed-data/train/*.tfrecord}"
EVAL_DATA="${EVAL_DATA:-/path-to-packed-data/eval/*.tfrecord}"
CKPT_DIR="${CKPT_DIR:-$REPO_ROOT/checkpoints}"
PHASE="${PHASE:-all}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --train_data) TRAIN_DATA="$2"; shift 2 ;;
    --eval_data)  EVAL_DATA="$2";  shift 2 ;;
    --ckpt_dir)   CKPT_DIR="$2";   shift 2 ;;
    --phase)      PHASE="$2";      shift 2 ;;
    *) echo "[ERROR] Bilinmeyen argüman: $1" >&2; exit 1 ;;
  esac
done

# ─── Ortak override ───────────────────────────────────────────────────────────
DATA_OVERRIDE="task.train_data.seq_length=128,\
task.train_data.max_predictions_per_seq=20,\
task.train_data.global_batch_size=16,\
task.validation_data.seq_length=128,\
task.validation_data.max_predictions_per_seq=20,\
task.validation_data.global_batch_size=16,\
task.train_data.input_path='$TRAIN_DATA',\
task.validation_data.input_path='$EVAL_DATA',\
trainer.validation_interval=1000,\
trainer.checkpoint_interval=5000,\
trainer.summary_interval=500"

PROG_K_OVERRIDE="task.model.encoder.any.token_keep_k1=96,\
task.model.encoder.any.token_keep_k2=64,\
task.model.encoder.any.token_keep_k3=32"

# ─── Aşama 1: 50,000 adım ─────────────────────────────────────────────────────
run_phase1() {
  echo ""
  echo "━━━ AŞAMA 1: 50k adım ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  local STEP_OVERRIDE="trainer.train_steps=50000,\
trainer.optimizer_config.warmup.polynomial.warmup_steps=2000,\
trainer.optimizer_config.learning_rate.polynomial.decay_steps=50000"

  echo "  Baseline (50k)..."
  $PYTHON "$TRAIN_SCRIPT" \
    --experiment=token_drop_bert/pretraining \
    --config_file="$REPO_ROOT/wiki_books_pretrain_sequence_pack.yaml" \
    --config_file="$REPO_ROOT/bert_en_uncased_base_token_drop.yaml" \
    --params_override="$DATA_OVERRIDE,$STEP_OVERRIDE" \
    --model_dir="$CKPT_DIR/short_baseline" \
    --mode=train_and_eval

  echo "  Progressive (50k)..."
  $PYTHON "$TRAIN_SCRIPT" \
    --experiment=progressive_drop_bert/pretraining \
    --config_file="$REPO_ROOT/wiki_books_pretrain_sequence_pack.yaml" \
    --config_file="$REPO_ROOT/experiments/progressive_contextual_dropping/bert_progressive_drop.yaml" \
    --params_override="$DATA_OVERRIDE,$STEP_OVERRIDE,$PROG_K_OVERRIDE" \
    --model_dir="$CKPT_DIR/short_progressive" \
    --mode=train_and_eval

  echo ""
  echo "Aşama 1 bitti. Go/No-Go kontrolü yapın:"
  echo "  python scripts/early_stop_monitor.py \\"
  echo "    --baseline_logdir=$CKPT_DIR/short_baseline/ \\"
  echo "    --progressive_logdir=$CKPT_DIR/short_progressive/"
  echo "  Ardından SST-2 fine-tune için: bash scripts/run_glue.sh --task SST-2 --steps 50000"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# ─── Aşama 2: 200,000 adım (50k checkpointtan devam) ─────────────────────────
run_phase2() {
  echo ""
  echo "━━━ AŞAMA 2: 200k adım (50k'dan devam) ━━━━━━━━━━━━━━━━━━━━"
  local STEP_OVERRIDE="trainer.train_steps=200000,\
trainer.optimizer_config.warmup.polynomial.warmup_steps=2000,\
trainer.optimizer_config.learning_rate.polynomial.decay_steps=200000"

  echo "  Baseline (50k → 200k)..."
  $PYTHON "$TRAIN_SCRIPT" \
    --experiment=token_drop_bert/pretraining \
    --config_file="$REPO_ROOT/wiki_books_pretrain_sequence_pack.yaml" \
    --config_file="$REPO_ROOT/bert_en_uncased_base_token_drop.yaml" \
    --params_override="$DATA_OVERRIDE,$STEP_OVERRIDE" \
    --model_dir="$CKPT_DIR/short_baseline" \
    --mode=train_and_eval

  echo "  Progressive (50k → 200k)..."
  $PYTHON "$TRAIN_SCRIPT" \
    --experiment=progressive_drop_bert/pretraining \
    --config_file="$REPO_ROOT/wiki_books_pretrain_sequence_pack.yaml" \
    --config_file="$REPO_ROOT/experiments/progressive_contextual_dropping/bert_progressive_drop.yaml" \
    --params_override="$DATA_OVERRIDE,$STEP_OVERRIDE,$PROG_K_OVERRIDE" \
    --model_dir="$CKPT_DIR/short_progressive" \
    --mode=train_and_eval

  echo ""
  echo "Aşama 2 bitti. Go/No-Go kontrolü yapın:"
  echo "  bash scripts/run_glue.sh --ckpt_steps 200000 --tasks SST-2,QNLI,MNLI,RTE"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

echo "============================================================"
echo " Kısa/Orta Koşu | Aşama: $PHASE | seq_len=128"
echo "============================================================"

case "$PHASE" in
  1|phase1) run_phase1 ;;
  2|phase2) run_phase2 ;;
  all)      run_phase1; run_phase2 ;;
  *) echo "[ERROR] --phase değeri 1, 2 veya all olmalı." >&2; exit 1 ;;
esac

echo ""
echo "============================================================"
echo " Kısa/Orta koşular tamamlandı."
echo " Sonuçları karşılaştırmak için:"
echo "   python analysis/compare_training_curves.py \\"
echo "     --baseline=$CKPT_DIR/short_baseline/ \\"
echo "     --progressive=$CKPT_DIR/short_progressive/"
echo "============================================================"
