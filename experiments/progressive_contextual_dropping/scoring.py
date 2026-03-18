# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pivot A — Alternatif token önem skoru fonksiyonları.

Eğer L2 norm tabanlı scoring aşama 0 pilot koşusunda başarısız olursa
(tüm tokenlar benzer skor üretiyor), bu modüldeki alternatif scoring
fonksiyonlarından biri encoder.py içindeki `_compute_drop_scores`
metoduna bağlanabilir.

Mevcut scoring seçenekleri:
  'l2_norm'        → Mevcut varsayılan (encoder.py içinde)
  'attention_cls'  → [CLS] attention ağırlıklarının çok-kafa ortalaması
  'layer_variance' → Hidden state boyutlar-arası varyans
  'gradient_norm'  → Hidden state gradyanının L2 normu (sadece eğitimde)

Kullanım (encoder.py call() içinde):
  from scoring import compute_drop_scores
  scores = compute_drop_scores(x, method=self._scoring_method, ...)
"""

import tensorflow as tf
tf_keras = tf.keras


def compute_drop_scores(
    hidden_states: tf.Tensor,
    method: str = "l2_norm",
    attention_weights: tf.Tensor = None,
    training: bool = False,
    tape: tf.GradientTape = None,
) -> tf.Tensor:
    """
    Token önem skorlarını hesapla.

    Args:
      hidden_states:    [B, T, H] — mevcut katman çıktısı.
      method:           Scoring yöntemi (bkz. modül docstring).
      attention_weights:[B, H, T, T] — self-attention ağırlıkları (attention_cls için).
      training:         True ise gradient_norm kullanılabilir.
      tape:             GradientTape nesnesi (gradient_norm için).

    Returns:
      scores: [B, T] — büyük değer = önemli token.
    """
    if method == "l2_norm":
        return _l2_norm_scores(hidden_states)
    elif method == "attention_cls":
        return _attention_cls_scores(hidden_states, attention_weights)
    elif method == "layer_variance":
        return _layer_variance_scores(hidden_states)
    elif method == "gradient_norm":
        return _gradient_norm_scores(hidden_states, tape)
    else:
        raise ValueError(
            f"Bilinmeyen scoring method: '{method}'. "
            f"Geçerli seçenekler: l2_norm, attention_cls, "
            f"layer_variance, gradient_norm"
        )


# ─── Yöntem 1: L2 Norm (varsayılan) ─────────────────────────────────────────

def _l2_norm_scores(hidden_states: tf.Tensor) -> tf.Tensor:
    """
    Her tokenin hidden state vektörünün L2 normu.

    Avantajlar: Sıfır ek parametre, hızlı, türevlenebilir.
    Dezavantajlar: LayerNorm sonrası normlar homojen olabilir.

    Returns: [B, T]
    """
    return tf.norm(hidden_states, axis=-1)


# ─── Yöntem 2: Attention-CLS Skoru ───────────────────────────────────────────

def _attention_cls_scores(
    hidden_states: tf.Tensor,
    attention_weights: tf.Tensor,
) -> tf.Tensor:
    """
    [CLS] tokenin her diğer tokene verdiği attention ağırlıklarını
    çok-kafa ortalaması olarak kullanır.

    Avantajlar: Semantik anlamlılıkla direkt ilgili.
    Dezavantajlar: Attention ağırlıklarına erişim gerektirir.

    Args:
      attention_weights: [B, num_heads, T, T] — self-attention ağırlıkları.

    Returns: [B, T]
    """
    if attention_weights is None:
        # Fallback: L2 norm
        tf.print(
            "[WARN] attention_cls için attention_weights gerekli, "
            "L2 norm kullanılıyor.", output_stream=sys.stderr
        )
        return _l2_norm_scores(hidden_states)

    # [CLS] indexi 0; attention_weights[B, H, 0, :] → [B, H, T]
    cls_attention = attention_weights[:, :, 0, :]  # [B, num_heads, T]
    # Kafalar üzerinden ortalama
    scores = tf.reduce_mean(cls_attention, axis=1)  # [B, T]
    return scores


# ─── Yöntem 3: Layer Variance ─────────────────────────────────────────────────

def _layer_variance_scores(hidden_states: tf.Tensor) -> tf.Tensor:
    """
    Her tokenin hidden state vektörünün boyutlar-arası varyansı.

    Motivasyon: LayerNorm sonrası L2 normlar homojenleşir, ama
    varyans bilgisi daha ayrışık olabilir.

    Avantajlar: L2 norm'a göre daha ayrışık olabilir.
    Dezavantajlar: Teorik motivasyon daha zayıf.

    Returns: [B, T]
    """
    # Boyutlar üzerinden varyans: [B, T]
    return tf.math.reduce_variance(hidden_states, axis=-1)


# ─── Yöntem 4: Gradient Norm ─────────────────────────────────────────────────

def _gradient_norm_scores(
    hidden_states: tf.Tensor,
    tape: tf.GradientTape,
) -> tf.Tensor:
    """
    Hidden state vektörünün gradyanının L2 normu.

    Motivasyon: Gradient norm, kayıp fonksiyonuna katkı sinyali taşır.
    Dezavantajlar: GradientTape gerektirir, sadece eğitimde kullanılabilir,
    bellek maliyeti yüksek.

    Args:
      tape: Aktif bir tf.GradientTape (hidden_states izlenmeli).

    Returns: [B, T]
    """
    if tape is None:
        # Inference'ta L2 norm'a geri dön
        return _l2_norm_scores(hidden_states)

    # dummy kayıp: hidden state normu (gerçek MLM kaybı dışarıdan verilmeli)
    dummy_loss = tf.reduce_sum(tf.norm(hidden_states, axis=-1))
    grads = tape.gradient(dummy_loss, hidden_states)  # [B, T, H]
    if grads is None:
        return _l2_norm_scores(hidden_states)
    return tf.norm(grads, axis=-1)  # [B, T]


# ─── Yardımcı: Scoring method factory ────────────────────────────────────────

SCORING_METHODS = {
    "l2_norm":       _l2_norm_scores,
    "attention_cls": _attention_cls_scores,
    "layer_variance": _layer_variance_scores,
    "gradient_norm": _gradient_norm_scores,
}

import sys


def list_methods() -> None:
    """Mevcut scoring yöntemlerini listele."""
    print("Mevcut token scoring yöntemleri:")
    descriptions = {
        "l2_norm":        "Hidden state L2 normu (varsayılan, önerilir)",
        "attention_cls":  "[CLS] attention ağırlıkları ortalaması",
        "layer_variance": "Hidden state boyutlar-arası varyans",
        "gradient_norm":  "Gradient L2 normu (sadece eğitim, yavaş)",
    }
    for method, desc in descriptions.items():
        print(f"  {method:<20} — {desc}")
