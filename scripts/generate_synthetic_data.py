#!/usr/bin/env python3
# =============================================================================
# generate_synthetic_data.py
#
# Wikipedia/BookCorpus indirmeden önce eğitim pipeline'ını test etmek için
# küçük bir sentetik TFRecord veri seti üretir.
#
# Çıktı: BERT pretraining formatında TFRecord dosyaları
# Format: BertPretrainDataConfig uyumlu (use_v2_feature_names=True)
#
# Kullanım:
#   python scripts/generate_synthetic_data.py \
#     --output_dir ./data/synthetic \
#     --seq_len 128 \
#     --n_train 1000 \
#     --n_eval 100
#
# Ardından train_pilot.sh içindeki input_path'i güncelleyin:
#   --train_data ./data/synthetic/train/*.tfrecord
#   --eval_data  ./data/synthetic/eval/*.tfrecord
# =============================================================================

import argparse
import os
import random

import numpy as np

try:
    import tensorflow as tf
except ImportError:
    print("[ERROR] tensorflow yüklü değil.", flush=True)
    raise

VOCAB_SIZE = 30522
SPECIAL_TOKENS = {
    "PAD": 0, "UNK": 100, "CLS": 101, "SEP": 102, "MASK": 103
}
# Normal kelime tokenleri: 104 - 30521
WORD_TOKEN_START = 104
WORD_TOKEN_END   = 30521


def random_segment(seq_len: int, vocab_size: int = VOCAB_SIZE,
                   rng: random.Random = None) -> list:
    """Rastgele bir token dizisi üret. [CLS] ile başlar, [SEP] ile biter."""
    if rng is None:
        rng = random.Random()
    content_len = seq_len - 2  # [CLS] + [SEP]
    tokens = [SPECIAL_TOKENS["CLS"]]
    tokens += [rng.randint(WORD_TOKEN_START, WORD_TOKEN_END)
               for _ in range(content_len)]
    tokens += [SPECIAL_TOKENS["SEP"]]
    return tokens


def apply_masking(token_ids: list, mask_prob: float = 0.15,
                  max_predictions: int = 20,
                  rng: random.Random = None):
    """
    BERT MLM maskeleme uygula.
    Döndürür:
      masked_ids      : [seq_len] — girdi (bazı tokenlar [MASK] ile değiştirildi)
      masked_lm_ids   : [max_predictions] — maskelenen tokenlerin gerçek kimlikleri
      masked_lm_pos   : [max_predictions] — maskelenen konumlar (0-padded)
      masked_lm_weights: [max_predictions] — gerçek vs. padding ağırlığı (1.0 veya 0.0)
    """
    if rng is None:
        rng = random.Random()

    seq_len = len(token_ids)
    # Özel tokenları maskelenemez yap
    mask_candidates = [
        i for i, t in enumerate(token_ids)
        if t not in SPECIAL_TOKENS.values()
    ]
    rng.shuffle(mask_candidates)
    n_mask = min(max_predictions, max(1, int(seq_len * mask_prob)))
    masked_positions = sorted(mask_candidates[:n_mask])

    masked_ids = token_ids[:]
    masked_lm_ids_list = []
    for pos in masked_positions:
        original = token_ids[pos]
        masked_lm_ids_list.append(original)
        r = rng.random()
        if r < 0.80:
            masked_ids[pos] = SPECIAL_TOKENS["MASK"]
        elif r < 0.90:
            masked_ids[pos] = rng.randint(WORD_TOKEN_START, WORD_TOKEN_END)
        # else: özgün token korunuyor

    # Padding
    n_real = len(masked_positions)
    pad_n  = max_predictions - n_real
    masked_positions   += [0] * pad_n
    masked_lm_ids_list += [0] * pad_n
    weights = [1.0] * n_real + [0.0] * pad_n

    return masked_ids, masked_lm_ids_list, masked_positions, weights


def make_example(seq_len: int, max_predictions: int, seed: int = 0) -> tf.train.Example:
    """
    Tek bir eğitim örneği oluştur (v2 feature names).
    """
    rng = random.Random(seed)
    token_ids = random_segment(seq_len, rng=rng)

    # Padding uygula (bu örnekte seq_len'e kadar zaten doldu)
    padded_ids = token_ids[:]

    masked_ids, masked_lm_ids, masked_lm_pos, masked_weights = apply_masking(
        token_ids, max_predictions=max_predictions, rng=rng
    )

    # Attention mask (1 = gerçek token, 0 = padding)
    attention_mask = [1] * len(masked_ids)

    # Token type ids (segment A = 0, B = 1 — burada sadece A kullanıyoruz)
    token_type_ids = [0] * seq_len

    def int_feature(vals):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=vals))

    def float_feature(vals):
        return tf.train.Feature(float_list=tf.train.FloatList(value=vals))

    # v2 feature names (BertPretrainDataConfig uyumlu)
    feature = {
        "input_word_ids":     int_feature(masked_ids),
        "input_mask":         int_feature(attention_mask),
        "input_type_ids":     int_feature(token_type_ids),
        "masked_lm_positions": int_feature(masked_lm_pos),
        "masked_lm_ids":      int_feature(masked_lm_ids),
        "masked_lm_weights":  float_feature(masked_weights),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def write_tfrecords(output_path: str, n_examples: int,
                    seq_len: int, max_predictions: int, seed_offset: int = 0):
    """n_examples örneği tek bir TFRecord dosyasına yaz."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with tf.io.TFRecordWriter(output_path) as writer:
        for i in range(n_examples):
            ex = make_example(seq_len, max_predictions, seed=seed_offset + i)
            writer.write(ex.SerializeToString())


def main():
    parser = argparse.ArgumentParser(
        description="Sentetik BERT pretraining verisi üretici"
    )
    parser.add_argument("--output_dir",    default="./data/synthetic",
                        help="TFRecord çıktı dizini")
    parser.add_argument("--seq_len",       type=int, default=128,
                        help="Dizi uzunluğu (varsayılan: 128)")
    parser.add_argument("--n_train",       type=int, default=2000,
                        help="Eğitim örnek sayısı (varsayılan: 2000)")
    parser.add_argument("--n_eval",        type=int, default=200,
                        help="Değerlendirme örnek sayısı (varsayılan: 200)")
    parser.add_argument("--n_train_shards", type=int, default=4,
                        help="Eğitim shard sayısı (varsayılan: 4)")
    parser.add_argument("--n_eval_shards",  type=int, default=1,
                        help="Değerlendirme shard sayısı (varsayılan: 1)")
    parser.add_argument("--mask_prob",     type=float, default=0.15)
    parser.add_argument("--seed",          type=int, default=42)
    args = parser.parse_args()

    max_pred = max(1, int(args.seq_len * args.mask_prob))
    # max_pred'i 20 ile sınırla (seq=128 için standart değer)
    max_pred = min(max_pred, 20)

    print(f"\n  Sentetik Veri Üretici")
    print(f"  seq_len      : {args.seq_len}")
    print(f"  max_pred     : {max_pred}")
    print(f"  n_train      : {args.n_train} ({args.n_train_shards} shard)")
    print(f"  n_eval       : {args.n_eval} ({args.n_eval_shards} shard)")
    print(f"  output_dir   : {args.output_dir}")
    print()

    import math

    # ── Eğitim shardları ───────────────────────────────────────────────────────
    train_dir = os.path.join(args.output_dir, "train")
    per_shard = math.ceil(args.n_train / args.n_train_shards)
    total_written = 0
    for s in range(args.n_train_shards):
        n = min(per_shard, args.n_train - s * per_shard)
        if n <= 0:
            break
        out_path = os.path.join(train_dir, f"pretrain_{s:04d}.tfrecord")
        write_tfrecords(out_path, n, args.seq_len, max_pred,
                        seed_offset=args.seed + s * per_shard)
        total_written += n
        print(f"  [train] Shard {s+1}/{args.n_train_shards}: "
              f"{n} ornek -> {os.path.basename(out_path)}")

    # ── Eval shardları ─────────────────────────────────────────────────────────
    eval_dir = os.path.join(args.output_dir, "eval")
    per_shard_eval = math.ceil(args.n_eval / args.n_eval_shards)
    for s in range(args.n_eval_shards):
        n = min(per_shard_eval, args.n_eval - s * per_shard_eval)
        if n <= 0:
            break
        out_path = os.path.join(eval_dir, f"pretrain_{s:04d}.tfrecord")
        write_tfrecords(out_path, n, args.seq_len, max_pred,
                        seed_offset=args.seed + 999999 + s * per_shard_eval)
        print(f"  [eval]  Shard {s+1}/{args.n_eval_shards}: "
              f"{n} ornek -> {os.path.basename(out_path)}")

    print()
    print(f"  Done! {total_written} train + {args.n_eval} eval examples written.")
    print()
    print("  To use in pilot run:")
    print(f"  bash scripts/train_pilot.sh \\")
    print(f"    --train_data {train_dir}/*.tfrecord \\")
    print(f"    --eval_data  {eval_dir}/*.tfrecord")


if __name__ == "__main__":
    main()
