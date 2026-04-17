#!/usr/bin/env bash
# =============================================================================
# Run #4 — Dynamic Masking Experiment
#
# 3-Way BERT comparison (Vanilla / TokenDrop / ProgDrop) with dynamic MLM
# masking: fresh masks applied every epoch instead of static pre-masked CSV.
#
# Architecture: BERT-base (hidden=768, layers=12, heads=12, inter=3072)
# Seq length:   512
# Data:         400K WikiText-103 samples (unmasked)
# =============================================================================

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
VENV="/data/umityilmaz/token_drop_l2/venv"
REPO="/data/umityilmaz/token_drop_v2/repo"
DATA_DIR="/data/umityilmaz/token_drop_v2/data"
CACHE_DIR="/data/umityilmaz/token_drop_l2/data/hf_cache"

UNMASKED_CSV="${DATA_DIR}/wikitext_unmasked_512.csv"
OUTPUT_DIR="/data/umityilmaz/token_drop_v2/checkpoints/run4_dynamic"

# ── Activate venv ─────────────────────────────────────────────────────────────
echo "Activating venv: ${VENV}"
source "${VENV}/bin/activate"
cd "${REPO}"

# ── Step 1: Prepare unmasked data (if not already done) ──────────────────────
if [ ! -f "${UNMASKED_CSV}" ]; then
    echo ""
    echo "============================================="
    echo "  Preparing unmasked tokenized data..."
    echo "============================================="
    python scripts/prepare_unmasked_data.py \
        --output_csv "${UNMASKED_CSV}" \
        --seq_len 512 \
        --max_samples 400000 \
        --dataset wikitext \
        --dataset_config wikitext-103-v1 \
        --split train \
        --cache_dir "${CACHE_DIR}"
    echo "Unmasked data ready: ${UNMASKED_CSV}"
else
    echo "Unmasked data already exists: ${UNMASKED_CSV}"
fi

# ── Step 2: Train with dynamic masking ────────────────────────────────────────
echo ""
echo "============================================="
echo "  Starting Run #4 — Dynamic Masking"
echo "  3 models: Vanilla, TokenDrop, ProgDrop"
echo "============================================="

python scripts/train_dynamic_masking.py \
    --data_path "${UNMASKED_CSV}" \
    --output_dir "${OUTPUT_DIR}" \
    --epochs 50000 \
    --max_steps 200000 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.06 \
    --early_stopping_patience 10 \
    --log_every 500 \
    --hidden_size 768 \
    --num_layers 12 \
    --num_heads 12 \
    --intermediate_size 3072 \
    --max_seq_len 512 \
    --token_keep_k 256 \
    --token_keep_k1 384 \
    --token_keep_k2 256 \
    --token_keep_k3 128 \
    --models vanilla tokendrop progressive

echo ""
echo "============================================="
echo "  Run #4 complete!"
echo "  Results: ${OUTPUT_DIR}"
echo "============================================="
