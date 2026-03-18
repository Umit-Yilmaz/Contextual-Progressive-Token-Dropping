#!/usr/bin/env bash
# =============================================================================
# prepare_data.sh
# Wikipedia + BookCorpus verisi indir ve BERT pretraining için TFRecord hazırla.
#
# Kullanım:
#   bash scripts/prepare_data.sh [--data_dir <dizin>] [--seq_len <uzunluk>]
#
# Çıktı:
#   <data_dir>/raw/           → ham metin dosyaları
#   <data_dir>/shards/        → tek bir Wikipedia + Books corpus metin
#   <data_dir>/tfrecords/     → paketlenmiş TFRecord dosyaları
#
# Gereksinimler:
#   pip install datasets apache-beam tensorflow-text
#   pip install wikiextractor (Wikipedia için)
#   tf-models-official (zaten tfenv içinde kurulu)
# =============================================================================

set -euo pipefail

# ─── Varsayılan parametreler ──────────────────────────────────────────────────
DATA_DIR="${DATA_DIR:-./data}"
SEQ_LEN="${SEQ_LEN:-512}"          # Pilot için 128, tam eğitim için 512
MAX_PRED="${MAX_PRED:-76}"         # seq_len=512 → 76; seq_len=128 → 20
NUM_TRAIN_SHARDS="${NUM_TRAIN_SHARDS:-256}"
NUM_EVAL_SHARDS="${NUM_EVAL_SHARDS:-8}"
VOCAB_FILE="${VOCAB_FILE:-./vocab.txt}"   # BERT uncased vocab
PYTHON="${PYTHON:-python}"
TFMODELS_DIR="${TFMODELS_DIR:-$(pip show tf-models-official 2>/dev/null | grep Location | awk '{print $2}')}"

# ─── Argüman ayrıştırma ───────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_dir)  DATA_DIR="$2";  shift 2 ;;
    --seq_len)   SEQ_LEN="$2";   shift 2 ;;
    --max_pred)  MAX_PRED="$2";  shift 2 ;;
    --vocab)     VOCAB_FILE="$2"; shift 2 ;;
    *) echo "[ERROR] Bilinmeyen argüman: $1" >&2; exit 1 ;;
  esac
done

echo "============================================================"
echo " BERT Pretraining Verisi Hazırlanıyor"
echo "  data_dir  : $DATA_DIR"
echo "  seq_len   : $SEQ_LEN"
echo "  max_pred  : $MAX_PRED"
echo "============================================================"

mkdir -p "$DATA_DIR/raw/wikipedia"
mkdir -p "$DATA_DIR/raw/bookcorpus"
mkdir -p "$DATA_DIR/shards"
mkdir -p "$DATA_DIR/tfrecords/train"
mkdir -p "$DATA_DIR/tfrecords/eval"

# ─── Adım 1: Vocab dosyasını indir ───────────────────────────────────────────
if [[ ! -f "$VOCAB_FILE" ]]; then
  echo "[1/5] BERT uncased vocab indiriliyor..."
  wget -q "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt" \
       -O "$VOCAB_FILE"
  echo "      vocab.txt indirildi."
else
  echo "[1/5] vocab.txt zaten mevcut, atlanıyor."
fi

# ─── Adım 2: Wikipedia indir ve çıkart ───────────────────────────────────────
WIKI_DUMP="$DATA_DIR/raw/wikipedia/enwiki-latest-pages-articles.xml.bz2"
if [[ ! -f "$DATA_DIR/raw/wikipedia/AA/wiki_00" ]]; then
  echo "[2/5] Wikipedia dump indiriliyor (bu uzun sürebilir: ~20GB)..."
  if [[ ! -f "$WIKI_DUMP" ]]; then
    wget -c "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2" \
         -O "$WIKI_DUMP"
  fi
  echo "      WikiExtractor çalıştırılıyor..."
  $PYTHON -m wikiextractor.WikiExtractor \
    "$WIKI_DUMP" \
    --output "$DATA_DIR/raw/wikipedia" \
    --bytes 100M \
    --processes 4 \
    --no-templates \
    --min_text_length 100
  echo "      Wikipedia çıkarıldı."
else
  echo "[2/5] Wikipedia zaten çıkarılmış, atlanıyor."
fi

# ─── Adım 3: BookCorpus → HuggingFace datasets ile ──────────────────────────
BOOKS_FILE="$DATA_DIR/raw/bookcorpus/books.txt"
if [[ ! -f "$BOOKS_FILE" ]]; then
  echo "[3/5] BookCorpus indiriliyor (HuggingFace datasets)..."
  $PYTHON - <<'EOF'
import sys
try:
    from datasets import load_dataset
except ImportError:
    print("[ERROR] 'datasets' paketi yok. Yükle: pip install datasets", file=sys.stderr)
    sys.exit(1)

import os
data_dir = os.environ.get("DATA_DIR", "./data")
out_path = f"{data_dir}/raw/bookcorpus/books.txt"
os.makedirs(os.path.dirname(out_path), exist_ok=True)

print("  BookCorpus yükleniyor (bookcorpusopen, ~1GB)...")
ds = load_dataset("bookcorpusopen", split="train", trust_remote_code=True)
print(f"  {len(ds):,} satır bulundu, yazılıyor...")
with open(out_path, "w", encoding="utf-8") as f:
    for ex in ds:
        text = ex["text"].strip()
        if text:
            f.write(text + "\n")
print(f"  Yazıldı: {out_path}")
EOF
  echo "      BookCorpus hazır."
else
  echo "[3/5] BookCorpus zaten mevcut, atlanıyor."
fi

# ─── Adım 4: Ham metinleri shardlara birleştir ───────────────────────────────
SHARD_TRAIN_DONE="$DATA_DIR/shards/.train_done"
if [[ ! -f "$SHARD_TRAIN_DONE" ]]; then
  echo "[4/5] Ham metinler shard dosyalarına bölünüyor..."
  $PYTHON - <<'EOF'
import os, glob, random, math

data_dir   = os.environ.get("DATA_DIR", "./data")
n_train    = int(os.environ.get("NUM_TRAIN_SHARDS", "256"))
n_eval     = int(os.environ.get("NUM_EVAL_SHARDS",  "8"))
shard_dir  = f"{data_dir}/shards"

# Tüm Wikipedia ve Books satırlarını topla
lines = []

print("  Wikipedia satırları okunuyor...")
for fp in glob.glob(f"{data_dir}/raw/wikipedia/**/*", recursive=True):
    if os.path.isfile(fp) and not fp.endswith(".bz2"):
        with open(fp, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("<"):
                    lines.append(line)

print(f"  Wikipedia: {len(lines):,} satır")

print("  BookCorpus satırları okunuyor...")
books_path = f"{data_dir}/raw/bookcorpus/books.txt"
if os.path.exists(books_path):
    with open(books_path, encoding="utf-8") as f:
        book_lines = [l.strip() for l in f if l.strip()]
    lines.extend(book_lines)
    print(f"  BookCorpus: {len(book_lines):,} satır")

random.seed(42)
random.shuffle(lines)
total = len(lines)
print(f"  Toplam: {total:,} satır")

# Eval split (%1)
n_eval_lines  = max(1000, total // 100)
eval_lines    = lines[:n_eval_lines]
train_lines   = lines[n_eval_lines:]

def write_shards(all_lines, n_shards, prefix):
    sz  = math.ceil(len(all_lines) / n_shards)
    for i in range(n_shards):
        chunk = all_lines[i*sz : (i+1)*sz]
        if not chunk:
            break
        fp = f"{shard_dir}/{prefix}_{i:04d}.txt"
        with open(fp, "w", encoding="utf-8") as f:
            f.write("\n".join(chunk))
    print(f"  {n_shards} {prefix} shard yazıldı.")

write_shards(train_lines, n_train, "train")
write_shards(eval_lines,  n_eval,  "eval")

open(f"{shard_dir}/.train_done", "w").close()
EOF
  echo "      Shardlama tamamlandı."
else
  echo "[4/5] Shardlar zaten mevcut, atlanıyor."
fi

# ─── Adım 5: TFRecord oluştur ─────────────────────────────────────────────────
TFRECORD_DONE="$DATA_DIR/tfrecords/.done_seq${SEQ_LEN}"
if [[ ! -f "$TFRECORD_DONE" ]]; then
  echo "[5/5] TFRecord dosyaları oluşturuluyor (seq_len=$SEQ_LEN)..."
  # tf-models-official içindeki create_pretraining_data.py kullanılır
  PRETRAIN_SCRIPT=$(find "$TFMODELS_DIR" -name "create_pretraining_data.py" 2>/dev/null | head -1)
  if [[ -z "$PRETRAIN_SCRIPT" ]]; then
    # Alternatif: doğrudan pip show ile bul
    PRETRAIN_SCRIPT=$($PYTHON -c \
      "import official; import os; print(os.path.dirname(official.__file__))"
    )/nlp/data/create_pretraining_data.py
  fi

  for SPLIT in train eval; do
    echo "  $SPLIT shardları işleniyor..."
    for shard in "$DATA_DIR/shards/${SPLIT}_"*.txt; do
      idx=$(basename "$shard" .txt | sed "s/${SPLIT}_//")
      out="$DATA_DIR/tfrecords/$SPLIT/pretrain_${idx}.tfrecord"
      if [[ ! -f "$out" ]]; then
        $PYTHON "$PRETRAIN_SCRIPT" \
          --input_file="$shard" \
          --output_file="$out" \
          --vocab_file="$VOCAB_FILE" \
          --do_lower_case=True \
          --max_seq_length="$SEQ_LEN" \
          --max_predictions_per_seq="$MAX_PRED" \
          --masked_lm_prob=0.15 \
          --random_seed=42 \
          --dupe_factor=10 2>/dev/null
      fi
    done
    echo "  $SPLIT TFRecord hazır."
  done
  touch "$TFRECORD_DONE"
  echo "      TFRecord oluşturma tamamlandı."
else
  echo "[5/5] TFRecord zaten mevcut (seq_len=$SEQ_LEN), atlanıyor."
fi

echo ""
echo "============================================================"
echo " Veri hazırlığı tamamlandı!"
echo ""
echo " Eğitimde kullanılacak yollar:"
echo "   train : $DATA_DIR/tfrecords/train/pretrain_*.tfrecord"
echo "   eval  : $DATA_DIR/tfrecords/eval/pretrain_*.tfrecord"
echo ""
echo " wiki_books_pretrain_sequence_pack.yaml içindeki input_path"
echo " alanlarını bu yollarla güncelleyin."
echo "============================================================"
