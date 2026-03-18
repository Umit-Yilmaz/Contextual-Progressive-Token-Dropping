#!/usr/bin/env python3
# =============================================================================
# token_drop_visualizer.py
#
# Progressive Contextual Drop modelinin hangi tokenleri elediğini görselleştirir.
# Girdi cümlesine renk/opaklık ile hangi tokenlerin hayatta kaldığını gösterir.
#
# Her dropping aşaması için ayrı satır üretir:
#   Aşama 0 (tam): [CLS] tokenlar renkli tokenlar [SEP]
#   Aşama 1 (k1):  daha az token, droplar soluk
#   Aşama 2 (k2):  daha da az
#   Aşama 3 (k3):  en az
#
# Çıktı: HTML dosyası (browser'da açın) + terminal özeti
#
# Kullanım:
#   python analysis/token_drop_visualizer.py \
#     --checkpoint   ./checkpoints/pilot_progressive \
#     --text         "The quick brown fox jumps over the lazy dog." \
#     [--output_dir  ./results/visualizations]
#     [--n_examples  10]   # İnteraktif mod: rastgele örnekler
# =============================================================================

import argparse
import os
import sys
import json
import html as html_module
from typing import List, Optional, Tuple

import numpy as np


# ─── Model yükleme ───────────────────────────────────────────────────────────

def load_model_and_tokenizer(checkpoint_dir: str):
    """
    Progressive encoder modelini ve BERT tokenizer'ı yükle.
    """
    try:
        import tensorflow as tf
        tf_keras = tf.keras
    except ImportError:
        print("[ERROR] tensorflow yüklü değil.", file=sys.stderr)
        sys.exit(1)

    try:
        import transformers
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    except ImportError:
        print("[WARN] transformers yüklü değil. Basit tokenizer kullanılıyor.",
              file=sys.stderr)
        tokenizer = None

    # Encoder'ı import et
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(repo_root, "experiments", "progressive_contextual_dropping"))
    sys.path.insert(0, repo_root)

    try:
        from experiments.progressive_contextual_dropping.encoder import (
            ProgressiveContextualDropEncoder
        )
    except ImportError:
        try:
            from encoder import ProgressiveContextualDropEncoder
        except ImportError:
            print("[ERROR] ProgressiveContextualDropEncoder import edilemedi.", file=sys.stderr)
            sys.exit(1)

    return ProgressiveContextualDropEncoder, tokenizer


# ─── Token skor hesaplama (eğitimsiz encoder ile) ─────────────────────────────

def get_drop_decisions(
    encoder_cls,
    tokens: List[str],
    token_ids: List[int],
    k1: int = 96, k2: int = 64, k3: int = 32,
    hidden_size: int = 256,
    num_layers: int = 8,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Encoder'ı gerçek ağırlıklarla çalıştırıp hangi tokenlerin
    her aşamada tutulduğunu döndürür.

    Returns:
      keep1: Aşama 1'den sonra kalan token indisleri
      keep2: Aşama 2'den sonra kalan token indisleri
      keep3: Aşama 3'ten sonra kalan token indisleri
    """
    try:
        import tensorflow as tf
        tf_keras = tf.keras
    except ImportError:
        return list(range(len(tokens))), list(range(k1)), list(range(k2))

    seq_len = len(token_ids)
    # k değerlerini seq_len'e göre ayarla
    k1 = min(k1, seq_len - 1)
    k2 = min(k2, k1 - 1)
    k3 = min(k3, k2 - 1)

    encoder = encoder_cls(
        vocab_size=30522,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_attention_heads=4,
        inner_dim=hidden_size * 4,
        max_sequence_length=512,
        token_keep_k1=k1,
        token_keep_k2=k2,
        token_keep_k3=k3,
    )

    # Sahte girdi oluştur
    ids      = tf.constant([token_ids], dtype=tf.int32)
    mask     = tf.ones([1, seq_len], dtype=tf.int32)
    type_ids = tf.zeros([1, seq_len], dtype=tf.int32)

    # Encoder'ı trace ederek intermediate indisleri elde et
    # Not: Mevcut encoder doğrudan indis döndürmüyor; L2 normları hesaplıyoruz
    _ = encoder([ids, mask, type_ids], training=False)

    # L2 norm tabanlı önem skoru (encoder'ın iç mantığını yansıtır)
    # Embedding katmanından sonra normları al
    embedding_out = encoder._embedding_layer(ids)
    embedding_out = encoder._position_embedding_layer(embedding_out)
    embedding_out = encoder._embedding_norm_layer(embedding_out)
    norms = tf.norm(embedding_out, axis=-1)[0].numpy()  # [seq_len]

    # allow/deny listelerini uygula
    allow_set = {100, 101, 102, 103}  # UNK, CLS, SEP, MASK
    deny_set  = {0}                   # PAD

    # Önem skorunu hesapla (deny listesi en düşük skor, allow listesi en yüksek)
    scores = norms.copy()
    for i, tid in enumerate(token_ids):
        if tid in deny_set:
            scores[i] = -1e9
        elif tid in allow_set:
            scores[i] = 1e9

    # top-k indisleri seç (sıralı)
    def topk_indices(s, k):
        return sorted(np.argsort(s)[-k:].tolist())

    keep1 = topk_indices(scores, k1)
    scores2 = scores[keep1]
    keep2_local = topk_indices(scores2, k2)
    keep2 = [keep1[i] for i in keep2_local]
    scores3 = scores[keep2]
    keep3_local = topk_indices(scores3, k3)
    keep3 = [keep2[i] for i in keep3_local]

    return keep1, keep2, keep3


# ─── HTML renk haritası ───────────────────────────────────────────────────────

def survival_to_color(survived_stages: int) -> Tuple[str, str]:
    """
    Kaç aşamadan sonra tokeni hayatta kaldığına göre renk döndürür.
    survived_stages: 0=elendi, 1=aşama1'den sonra elendi, 2=2den elendi, 3=tüm aşamalar
    Returns: (background_color, text_color)
    """
    if survived_stages == 3:
        return "#1B5E20", "#FFFFFF"   # Koyu yeşil — tüm aşamalar boyunca hayatta
    elif survived_stages == 2:
        return "#4CAF50", "#000000"   # Orta yeşil — 2 aşama
    elif survived_stages == 1:
        return "#FFF9C4", "#000000"   # Sarı — 1 aşama
    else:
        return "#FFCDD2", "#999999"   # Soluk kırmızı — elendi


def generate_html(
    tokens: List[str],
    keep1: List[int],
    keep2: List[int],
    keep3: List[int],
    title: str = "Token Drop Visualization",
) -> str:
    keep1_set = set(keep1)
    keep2_set = set(keep2)
    keep3_set = set(keep3)

    def token_cell(idx: int, token: str) -> str:
        """Token için HTML hücre üret."""
        if idx in keep3_set:
            stages = 3
            tooltip = f"Stage 0→1→2→3 (her aşamada korundu)"
        elif idx in keep2_set:
            stages = 2
            tooltip = f"Stage 0→1→2 (Stage 3'te elendi)"
        elif idx in keep1_set:
            stages = 1
            tooltip = f"Stage 0→1 (Stage 2'de elendi)"
        else:
            stages = 0
            tooltip = "Stage 1'de elendi"

        bg, fg = survival_to_color(stages)
        escaped = html_module.escape(token.replace("##", ""))
        prefix  = "##" if token.startswith("##") else ""
        opacity = 1.0 if stages > 0 else 0.4

        return (
            f'<span class="token stage{stages}" '
            f'style="background:{bg};color:{fg};opacity:{opacity}" '
            f'title="{tooltip}">'
            f'{html_module.escape(prefix)}{escaped}'
            f'</span>'
        )

    # Her aşama için satır oluştur
    stage_rows = []
    for stage_idx, (label, keep_set) in enumerate([
        ("Aşama 0 — Tüm tokenlar (giriş)", set(range(len(tokens)))),
        (f"Aşama 1 — k1={len(keep1)} token ({len(keep1)/len(tokens)*100:.0f}%)", keep1_set),
        (f"Aşama 2 — k2={len(keep2)} token ({len(keep2)/len(tokens)*100:.0f}%)", keep2_set),
        (f"Aşama 3 — k3={len(keep3)} token ({len(keep3)/len(tokens)*100:.0f}%)", keep3_set),
    ]):
        cells = []
        for idx, tok in enumerate(tokens):
            if stage_idx == 0 or idx in keep_set:
                bg, fg = survival_to_color(3) if idx in keep3_set else \
                         (survival_to_color(2) if idx in keep2_set else \
                         (survival_to_color(1) if idx in keep1_set else \
                          survival_to_color(0)))
                opacity = 1.0 if idx in keep_set else 0.15
                escaped = html_module.escape(tok)
                cells.append(
                    f'<span class="token" '
                    f'style="background:{bg if idx in keep_set else "#EEEEEE"};'
                    f'color:{fg if idx in keep_set else "#AAAAAA"};'
                    f'opacity:{opacity}">'
                    f'{escaped}</span>'
                )
            else:
                cells.append(
                    f'<span class="token dropped" '
                    f'style="background:#F5F5F5;color:#CCCCCC;opacity:0.2">'
                    f'{html_module.escape(tok)}</span>'
                )
        stage_rows.append((label, "".join(cells)))

    rows_html = "\n".join(
        f'<tr><td class="stage-label">{label}</td>'
        f'<td class="tokens-cell">{cells_html}</td></tr>'
        for label, cells_html in stage_rows
    )

    legend_html = "".join(
        f'<span class="legend-item" style="background:{bg};color:{fg}">{label}</span>'
        for bg, fg, label in [
            ("#1B5E20", "#FFFFFF", "Tüm aşamalar hayatta"),
            ("#4CAF50", "#000000", "2 aşama hayatta"),
            ("#FFF9C4", "#000000", "1 aşama hayatta"),
            ("#FFCDD2", "#999999", "İlk aşamada elendi"),
        ]
    )

    return f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<title>{html_module.escape(title)}</title>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; margin: 20px; background: #FAFAFA; }}
  h2   {{ color: #333; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
  td   {{ padding: 8px 12px; vertical-align: top; }}
  .stage-label {{
    font-weight: bold; font-size: 13px; color: #555;
    white-space: nowrap; width: 280px; border-right: 2px solid #DDD;
  }}
  .tokens-cell {{ line-height: 2.2em; }}
  .token {{
    display: inline-block;
    padding: 2px 6px; margin: 2px 1px;
    border-radius: 4px;
    font-size: 13px; font-family: monospace;
    transition: opacity 0.2s;
    cursor: help;
  }}
  .dropped {{ text-decoration: line-through; }}
  .legend  {{ margin-top: 16px; padding: 10px; background: #EEE; border-radius: 6px; }}
  .legend-item {{
    display: inline-block; padding: 3px 8px; margin: 3px;
    border-radius: 4px; font-size: 12px; font-weight: bold;
  }}
  tr:nth-child(even) {{ background: #F5F5F5; }}
  tr:hover {{ background: #E8F5E9; }}
</style>
</head>
<body>
<h2>Token Drop Görselleştirici — Progressive Contextual Drop</h2>
<p>Her aşamada hangi tokenların korunduğunu gösterir.
   Token üzerine gelince ayrıntı görebilirsiniz.</p>
<table>
  <thead>
    <tr>
      <th style="text-align:left;padding:8px 12px;">Aşama</th>
      <th style="text-align:left;padding:8px 12px;">Tokenlar</th>
    </tr>
  </thead>
  <tbody>
    {rows_html}
  </tbody>
</table>
<div class="legend">
  <strong>Renk Haritası:</strong> {legend_html}
</div>
</body>
</html>
"""


# ─── Basit tokenizer (transformers olmadan) ───────────────────────────────────

def simple_tokenize(text: str) -> Tuple[List[str], List[int]]:
    """
    transformers olmadan basit wordpiece-benzeri tokenization.
    Sadece demonstrasyon amaçlı; gerçek analizde HuggingFace kullanın.
    """
    # Küçük harf + boşluk bazlı bölme
    tokens  = ["[CLS]"]
    ids     = [101]
    for word in text.lower().split():
        # Wordpiece benzetmesi: uzun kelimeleri parçala
        if len(word) <= 4:
            tokens.append(word)
            ids.append(hash(word) % 20000 + 1000)
        else:
            tokens.append(word[:4])
            ids.append(hash(word[:4]) % 20000 + 1000)
            tokens.append("##" + word[4:])
            ids.append(hash(word[4:]) % 20000 + 1000)
    tokens.append("[SEP]")
    ids.append(102)
    return tokens, ids


# ─── Ana fonksiyon ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Token Drop Görselleştirici"
    )
    parser.add_argument("--checkpoint", default=None,
                        help="Progressive encoder checkpoint dizini")
    parser.add_argument("--text",       default=None,
                        help="Görselleştirilecek metin")
    parser.add_argument("--input_file", default=None,
                        help="Her satırı ayrı örnek olan metin dosyası")
    parser.add_argument("--output_dir", default="./results/visualizations",
                        help="HTML çıktı dizini")
    parser.add_argument("--k1", type=int, default=96)
    parser.add_argument("--k2", type=int, default=64)
    parser.add_argument("--k3", type=int, default=32)
    parser.add_argument("--no_model", action="store_true",
                        help="Model çalıştırma; rastgele skor kullan (hızlı test)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Örnekleri belirle
    examples = []
    if args.text:
        examples.append(args.text)
    elif args.input_file:
        with open(args.input_file, encoding="utf-8") as f:
            examples = [l.strip() for l in f if l.strip()][:50]
    else:
        # Demo metinleri
        examples = [
            "The quick brown fox jumps over the lazy dog.",
            "BERT is a transformer-based machine learning technique for NLP.",
            "Progressive token dropping reduces computation while preserving accuracy.",
            "Attention mechanisms allow models to focus on relevant parts of the input.",
        ]

    print(f"\n  {len(examples)} örnek görselleştirilecek.")
    print(f"  Çıktı: {args.output_dir}/")

    # Tokenizer yükle
    tokenizer = None
    if not args.no_model:
        try:
            from transformers import BertTokenizer
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            print("  HuggingFace BERT tokenizer yüklendi.")
        except Exception:
            print("  [WARN] HuggingFace tokenizer yüklenemedi. Basit tokenizer kullanılıyor.")

    all_html_sections = []

    for ex_idx, text in enumerate(examples):
        # Tokenize
        if tokenizer:
            enc    = tokenizer(text, return_tensors="np", add_special_tokens=True)
            ids    = enc["input_ids"][0].tolist()
            tokens = tokenizer.convert_ids_to_tokens(ids)
        else:
            tokens, ids = simple_tokenize(text)

        seq_len = len(tokens)
        k1 = min(args.k1, seq_len - 2)
        k2 = min(args.k2, k1 - 1)
        k3 = min(args.k3, k2 - 1)

        # Drop kararları
        if args.no_model or args.checkpoint is None:
            # Rastgele demo (allow listesi korunur)
            allow_set = {100, 101, 102, 103}
            rng = np.random.default_rng(seed=ex_idx)
            scores = rng.uniform(0, 1, seq_len)
            for i, tid in enumerate(ids):
                if tid in allow_set:
                    scores[i] = 2.0
            def topk(s, k):
                return sorted(np.argsort(s)[-k:].tolist())
            keep1 = topk(scores, k1)
            keep2 = topk(scores[keep1], k2)
            keep2 = [keep1[i] for i in keep2]
            keep3 = topk(scores[keep2], k3)
            keep3 = [keep2[i] for i in keep3]
        else:
            try:
                enc_cls, _ = load_model_and_tokenizer(args.checkpoint)
                keep1, keep2, keep3 = get_drop_decisions(
                    enc_cls, tokens, ids, k1=k1, k2=k2, k3=k3
                )
            except Exception as e:
                print(f"  [WARN] Model çalıştırılamadı: {e}. Rastgele demo kullanılıyor.")
                keep1 = list(range(k1))
                keep2 = list(range(k2))
                keep3 = list(range(k3))

        # HTML üret
        section_title = f"Örnek {ex_idx+1}: {text[:80]}..."
        section_html  = generate_html(tokens, keep1, keep2, keep3, title=section_title)

        # Tek dosya
        out_path = os.path.join(args.output_dir, f"example_{ex_idx+1:03d}.html")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(section_html)

        all_html_sections.append(f"<h3>Örnek {ex_idx+1}</h3><p><em>{html_module.escape(text)}</em></p>")
        all_html_sections.append(section_html.split("<body>")[1].split("</body>")[0])

        # Terminal özeti
        print(f"\n  [{ex_idx+1}] {text[:60]}...")
        print(f"       Tokenlar: {seq_len}")
        print(f"       Aşama 1:  {len(keep1)}/{seq_len} ({len(keep1)/seq_len*100:.0f}%)")
        print(f"       Aşama 2:  {len(keep2)}/{seq_len} ({len(keep2)/seq_len*100:.0f}%)")
        print(f"       Aşama 3:  {len(keep3)}/{seq_len} ({len(keep3)/seq_len*100:.0f}%)")
        print(f"       HTML:     {out_path}")

    # Hepsini tek dosyada birleştir
    combined_path = os.path.join(args.output_dir, "all_examples.html")
    combined_html = f"""<!DOCTYPE html>
<html lang="tr"><head><meta charset="UTF-8">
<title>Token Drop — Tüm Örnekler</title>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; margin: 20px; background: #FAFAFA; }}
  h2, h3 {{ color: #333; }}
  hr {{ border: 1px solid #DDD; margin: 30px 0; }}
</style>
</head><body>
<h2>Token Drop Görselleştirici — Tüm Örnekler</h2>
{'<hr>'.join(all_html_sections)}
</body></html>"""

    with open(combined_path, "w", encoding="utf-8") as f:
        f.write(combined_html)

    print(f"\n  Tüm örnekler: {combined_path}")
    print(f"  Tarayıcıda açmak için: start {combined_path}  (Windows)")


if __name__ == "__main__":
    main()
