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

"""Progressive Contextual Token Dropping BERT encoder.

Unlike the baseline TokenDropBertEncoder which uses a static vocabulary-level
moving average of MLM loss to decide which tokens to drop (decided BEFORE any
transformer layer runs), this encoder scores tokens using the L2 norm of their
actual hidden-state representations at three progressive drop points throughout
the network.

Key differences from baseline:
  - Scoring is context-aware: computed from live hidden states, not a lookup table.
  - Dropping is progressive: 3 stages instead of 1.
  - No extra trainable/non-trainable parameters for importance tracking.
  - Cold-start ready: contextual scoring is valid from step 1 of training.
"""

from typing import Any, Callable, List, Optional, Tuple, Union

import tensorflow as tf
tf_keras = tf.keras

from official.modeling import tf_utils
from official.nlp.modeling import layers

_Initializer = Union[str, tf_keras.initializers.Initializer]
_Activation = Union[str, Callable[..., Any]]

_approx_gelu = lambda x: tf_keras.activations.gelu(x, approximate=True)


class ProgressiveContextualDropEncoder(tf_keras.layers.Layer):
  """BERT encoder with three-stage progressive contextual token dropping.

  Drops tokens at three equidistant points in the network by scoring them
  using the L2 norm of their current hidden-state representations.
  Tokens on the allow-list (special tokens) are never dropped; tokens on
  the deny-list (padding) are always dropped.

  For a 12-layer BERT with default settings:

    Layers  0- 2  (3 layers): all N tokens
    Drop  1:                   keep top-k1 by ||h||_2   (N   -> k1)
    Layers  3- 5  (3 layers): k1 tokens
    Drop  2:                   keep top-k2 by ||h||_2   (k1  -> k2)
    Layers  6- 8  (3 layers): k2 tokens
    Drop  3:                   keep top-k3 by ||h||_2   (k2  -> k3)
    Layers  9-10  (2 layers): k3 tokens
    Layer  11     (1 layer):  all N tokens  (dropped tokens reinserted)

  Dropped tokens are frozen at their last computed hidden state and reinserted
  at their original positions before the final transformer layer, so the
  output is always a full-length sequence — identical interface to standard BERT.

  Args:
    vocab_size: Size of token vocabulary.
    hidden_size: Transformer hidden dimension.
    num_layers: Number of transformer layers. Must be >= 4.
    num_attention_heads: Number of self-attention heads.
    max_sequence_length: Maximum input sequence length.
    type_vocab_size: Number of token-type (segment) IDs.
    inner_dim: FFN inner (intermediate) dimension.
    inner_activation: Activation function for FFN inner layer.
    output_dropout: Dropout rate after attention and FFN.
    attention_dropout: Dropout rate on attention weights.
    token_keep_k1: Tokens retained after drop stage 1. Default 384.
    token_keep_k2: Tokens retained after drop stage 2. Default 256.
    token_keep_k3: Tokens retained after drop stage 3. Default 128.
      Must satisfy: token_keep_k3 < token_keep_k2 < token_keep_k1 < seq_len.
    token_allow_list: Token IDs that are NEVER dropped (e.g. [CLS], [SEP],
      [MASK], [UNK]). Their scores receive a +1e4 offset.
    token_deny_list: Token IDs that are ALWAYS dropped (e.g. [PAD]).
      Their scores receive a -1e4 offset.
    initializer: Weight initializer for all sub-layers.
    output_range: If set, only the first `output_range` positions of the final
      layer are returned. None returns the full sequence.
    embedding_width: Width of word embeddings. If != hidden_size, a projection
      matrix is added.
    norm_first: If True, apply LayerNorm before attention/FFN (Pre-LN).
  """

  def __init__(
      self,
      vocab_size: int,
      hidden_size: int = 768,
      num_layers: int = 12,
      num_attention_heads: int = 12,
      max_sequence_length: int = 512,
      type_vocab_size: int = 16,
      inner_dim: int = 3072,
      inner_activation: _Activation = _approx_gelu,
      output_dropout: float = 0.1,
      attention_dropout: float = 0.1,
      token_keep_k1: int = 384,
      token_keep_k2: int = 256,
      token_keep_k3: int = 128,
      token_allow_list: Tuple[int, ...] = (100, 101, 102, 103),
      token_deny_list: Tuple[int, ...] = (0,),
      initializer: _Initializer = tf_keras.initializers.TruncatedNormal(
          stddev=0.02),
      output_range: Optional[int] = None,
      embedding_width: Optional[int] = None,
      norm_first: bool = False,
      **kwargs):
    # Pop legacy kwargs forwarded from BertEncoderConfig for compatibility.
    for _key in ('dict_outputs', 'return_all_encoder_outputs'):
      kwargs.pop(_key, None)
    if 'intermediate_size' in kwargs:
      inner_dim = kwargs.pop('intermediate_size')
    if 'activation' in kwargs:
      inner_activation = kwargs.pop('activation')
    if 'dropout_rate' in kwargs:
      output_dropout = kwargs.pop('dropout_rate')
    if 'attention_dropout_rate' in kwargs:
      attention_dropout = kwargs.pop('attention_dropout_rate')

    super().__init__(**kwargs)

    if num_layers < 4:
      raise ValueError(
          f'num_layers must be >= 4 for progressive dropping, got {num_layers}.')
    if not (token_keep_k3 < token_keep_k2 < token_keep_k1):
      raise ValueError(
          f'Require token_keep_k3 < token_keep_k2 < token_keep_k1, '
          f'got {token_keep_k3} < {token_keep_k2} < {token_keep_k1}.')
    if token_keep_k1 >= max_sequence_length:
      raise ValueError(
          f'token_keep_k1 ({token_keep_k1}) must be less than '
          f'max_sequence_length ({max_sequence_length}).')

    self._token_allow_list = token_allow_list
    self._token_deny_list = token_deny_list
    self._k1 = token_keep_k1
    self._k2 = token_keep_k2
    self._k3 = token_keep_k3
    self._num_layers = num_layers

    activation = tf_keras.activations.get(inner_activation)
    initializer = tf_keras.initializers.get(initializer)

    if embedding_width is None:
      embedding_width = hidden_size

    self._embedding_layer = layers.OnDeviceEmbedding(
        vocab_size=vocab_size,
        embedding_width=embedding_width,
        initializer=tf_utils.clone_initializer(initializer),
        name='word_embeddings')

    self._position_embedding_layer = layers.PositionEmbedding(
        initializer=tf_utils.clone_initializer(initializer),
        max_length=max_sequence_length,
        name='position_embedding')

    self._type_embedding_layer = layers.OnDeviceEmbedding(
        vocab_size=type_vocab_size,
        embedding_width=embedding_width,
        initializer=tf_utils.clone_initializer(initializer),
        use_one_hot=True,
        name='type_embeddings')

    self._embedding_norm_layer = tf_keras.layers.LayerNormalization(
        name='embeddings_layer_norm', axis=-1, epsilon=1e-12, dtype=tf.float32)

    self._embedding_dropout = tf_keras.layers.Dropout(
        rate=output_dropout, name='embedding_dropout')

    self._embedding_projection = None
    if embedding_width != hidden_size:
      self._embedding_projection = tf_keras.layers.EinsumDense(
          '...x,xy->...y',
          output_shape=hidden_size,
          bias_axes='y',
          kernel_initializer=tf_utils.clone_initializer(initializer),
          name='embedding_projection')

    self._attention_mask_layer = layers.SelfAttentionMask(
        name='self_attention_mask')

    self._transformer_layers = []
    for i in range(num_layers):
      layer = layers.TransformerEncoderBlock(
          num_attention_heads=num_attention_heads,
          inner_dim=inner_dim,
          inner_activation=inner_activation,
          output_dropout=output_dropout,
          attention_dropout=attention_dropout,
          norm_first=norm_first,
          kernel_initializer=tf_utils.clone_initializer(initializer),
          name='transformer_layer_%d' % i)
      self._transformer_layers.append(layer)

    self._pooler_layer = tf_keras.layers.Dense(
        units=hidden_size,
        activation='tanh',
        kernel_initializer=tf_utils.clone_initializer(initializer),
        name='pooler_transform')

    self._config = {
        'vocab_size': vocab_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'num_attention_heads': num_attention_heads,
        'max_sequence_length': max_sequence_length,
        'type_vocab_size': type_vocab_size,
        'inner_dim': inner_dim,
        'inner_activation': tf_keras.activations.serialize(activation),
        'output_dropout': output_dropout,
        'attention_dropout': attention_dropout,
        'token_keep_k1': token_keep_k1,
        'token_keep_k2': token_keep_k2,
        'token_keep_k3': token_keep_k3,
        'token_allow_list': token_allow_list,
        'token_deny_list': token_deny_list,
        'initializer': tf_keras.initializers.serialize(initializer),
        'output_range': output_range,
        'embedding_width': embedding_width,
        'norm_first': norm_first,
    }

    self.inputs = dict(
        input_word_ids=tf_keras.Input(shape=(None,), dtype=tf.int32),
        input_mask=tf_keras.Input(shape=(None,), dtype=tf.int32),
        input_type_ids=tf_keras.Input(shape=(None,), dtype=tf.int32))

  # ── Scoring ──────────────────────────────────────────────────────────────────

  def _compute_drop_scores(
      self, hidden: tf.Tensor, word_ids: tf.Tensor) -> tf.Tensor:
    """Score tokens by L2 norm of hidden state, with allow/deny overrides.

    Args:
      hidden:   [batch, seq_len, hidden_size]  float tensor.
      word_ids: [batch, seq_len]               int32 tensor of token IDs.

    Returns:
      scores: [batch, seq_len]  Higher score = more important = kept longer.
    """
    # Geometric importance: how much information the representation carries.
    scores = tf.norm(tf.cast(hidden, tf.float32), axis=-1)  # [B, S]

    # Hard overrides via additive offsets so allow/deny dominate any norm value.
    for tid in self._token_allow_list:
      scores = scores + tf.cast(tf.equal(word_ids, tid), tf.float32) * 1e4
    for tid in self._token_deny_list:
      scores = scores + tf.cast(tf.equal(word_ids, tid), tf.float32) * (-1e4)

    return scores

  def _split_topk(
      self, scores: tf.Tensor, k: int) -> Tuple[tf.Tensor, tf.Tensor]:
    """Partition token indices into top-k and the rest, both in ascending order.

    Args:
      scores: [batch, seq_len]  Importance scores.
      k:      Number of tokens to keep.

    Returns:
      selected:     [batch, k]           Indices of top-k tokens.
      not_selected: [batch, seq_len - k] Indices of the remaining tokens.
    """
    # Rank all positions descending by score.
    ranked = tf.argsort(scores, axis=-1, direction='DESCENDING')  # [B, S]
    # Keep positions sorted ascending so gather preserves left-to-right order.
    selected = tf.sort(ranked[:, :k], axis=-1)
    not_selected = tf.sort(ranked[:, k:], axis=-1)
    return selected, not_selected

  # ── Forward pass ─────────────────────────────────────────────────────────────

  def call(self, inputs, output_range: Optional[tf.Tensor] = None):
    if not isinstance(inputs, dict):
      raise ValueError('Unexpected inputs type to %s.' % self.__class__)

    word_ids  = inputs['input_word_ids']   # [B, N]
    mask      = inputs['input_mask']       # [B, N]
    type_ids  = inputs['input_type_ids']   # [B, N]

    # ── Embeddings ─────────────────────────────────────────────────────────────
    word_embeddings     = self._embedding_layer(word_ids)
    position_embeddings = self._position_embedding_layer(word_embeddings)
    type_embeddings     = self._type_embedding_layer(type_ids)

    embeddings = word_embeddings + position_embeddings + type_embeddings
    embeddings = self._embedding_norm_layer(embeddings)
    embeddings = self._embedding_dropout(embeddings)

    if self._embedding_projection is not None:
      embeddings = self._embedding_projection(embeddings)

    # ── Routing state ──────────────────────────────────────────────────────────
    x              = embeddings             # [B, N, H]  — live hidden states
    live_word_ids  = word_ids              # [B, N]     — IDs for live tokens
    live_mask      = mask                  # [B, N]     — padding mask for live

    batch_size = tf.shape(word_ids)[0]
    seq_len    = tf.shape(word_ids)[1]

    # Runtime guard: k1 must be strictly less than the actual sequence length.
    tf.debugging.assert_less(
        tf.cast(self._k1, tf.int32),
        tf.cast(seq_len, tf.int32),
        message=(f'token_keep_k1 ({self._k1}) must be less than the '
                 f'runtime sequence length. Reduce token_keep_k1 or '
                 f'increase the sequence length.'))

    # live_indices[b, i] = original position of the i-th live token in batch b.
    live_indices = tf.tile(
        tf.expand_dims(tf.range(seq_len), 0), [batch_size, 1])  # [B, N]

    # Buckets for frozen tokens (appended in drop order).
    frozen_states: List[tf.Tensor]  = []
    frozen_idx:    List[tf.Tensor]  = []

    attention_mask = self._attention_mask_layer(x, live_mask)
    encoder_outputs = []
    n = self._num_layers

    # ── Stage 0: all tokens, first n//4 layers ─────────────────────────────────
    for layer in self._transformer_layers[:n // 4]:
      x = layer([x, attention_mask])
      encoder_outputs.append(x)

    # ── Drop 1: N → k1  (score by ||h||₂ after stage-0 context) ───────────────
    scores1        = self._compute_drop_scores(x, live_word_ids)
    keep1, drop1   = self._split_topk(scores1, self._k1)

    frozen_states.append(tf.gather(x,            drop1, batch_dims=1))
    frozen_idx.append(   tf.gather(live_indices, drop1, batch_dims=1))

    x             = tf.gather(x,            keep1, batch_dims=1)
    live_word_ids = tf.gather(live_word_ids, keep1, batch_dims=1)
    live_mask     = tf.gather(live_mask,     keep1, batch_dims=1)
    live_indices  = tf.gather(live_indices,  keep1, batch_dims=1)
    attention_mask = self._attention_mask_layer(x, live_mask)

    # ── Stage 1: k1 tokens, next n//4 layers ──────────────────────────────────
    # NOTE: intermediate outputs are NOT appended to encoder_outputs here because
    # they have shape [B, k1, H] — incompatible with the full-length [B, N, H]
    # tensors produced by Stage 0 and the final layer. Appending mixed shapes
    # would break any downstream code that stacks or concatenates encoder_outputs.
    for layer in self._transformer_layers[n // 4: n // 2]:
      x = layer([x, attention_mask])

    # ── Drop 2: k1 → k2  (score by ||h||₂ after stage-1 context) ──────────────
    scores2        = self._compute_drop_scores(x, live_word_ids)
    keep2, drop2   = self._split_topk(scores2, self._k2)

    frozen_states.append(tf.gather(x,            drop2, batch_dims=1))
    frozen_idx.append(   tf.gather(live_indices, drop2, batch_dims=1))

    x             = tf.gather(x,            keep2, batch_dims=1)
    live_word_ids = tf.gather(live_word_ids, keep2, batch_dims=1)
    live_mask     = tf.gather(live_mask,     keep2, batch_dims=1)
    live_indices  = tf.gather(live_indices,  keep2, batch_dims=1)
    attention_mask = self._attention_mask_layer(x, live_mask)

    # ── Stage 2: k2 tokens, next n//4 layers ──────────────────────────────────
    # (see Stage 1 note — intermediate outputs are not appended)
    for layer in self._transformer_layers[n // 2: 3 * n // 4]:
      x = layer([x, attention_mask])

    # ── Drop 3: k2 → k3  (score by ||h||₂ after stage-2 context) ──────────────
    scores3        = self._compute_drop_scores(x, live_word_ids)
    keep3, drop3   = self._split_topk(scores3, self._k3)

    frozen_states.append(tf.gather(x,            drop3, batch_dims=1))
    frozen_idx.append(   tf.gather(live_indices, drop3, batch_dims=1))

    x             = tf.gather(x,             keep3, batch_dims=1)
    live_word_ids = tf.gather(live_word_ids, keep3, batch_dims=1)
    live_mask     = tf.gather(live_mask,     keep3, batch_dims=1)
    live_indices  = tf.gather(live_indices,  keep3, batch_dims=1)
    attention_mask = self._attention_mask_layer(x, live_mask)

    # ── Stage 3: k3 tokens, remaining layers except the final one ─────────────
    # (see Stage 1 note — intermediate outputs are not appended)
    for layer in self._transformer_layers[3 * n // 4: -1]:
      x = layer([x, attention_mask])

    # ── Reintegrate: restore all tokens to original positions ─────────────────
    # Concatenate live tokens with all frozen buckets (reverse drop order so
    # the index arithmetic is transparent, but any order works).
    all_x = tf.concat(
        [x] + [tf.cast(fs, x.dtype) for fs in reversed(frozen_states)],
        axis=1)  # [B, N, H]
    all_indices = tf.concat(
        [live_indices] + list(reversed(frozen_idx)),
        axis=1)  # [B, N]

    # Invert the permutation to restore original token ordering.
    reverse_perm = tf.argsort(all_indices, axis=-1)
    x = tf.gather(all_x, reverse_perm, batch_dims=1)  # [B, N, H]

    # ── Final layer: full sequence attends to full sequence ────────────────────
    full_attention_mask = self._attention_mask_layer(x, mask)
    x = self._transformer_layers[-1](
        [x, full_attention_mask],
        output_range=output_range)
    encoder_outputs.append(x)

    last_encoder_output = encoder_outputs[-1]
    first_token_tensor  = last_encoder_output[:, 0, :]
    pooled_output       = self._pooler_layer(first_token_tensor)

    return dict(
        sequence_output=encoder_outputs[-1],
        pooled_output=pooled_output,
        encoder_outputs=encoder_outputs)

  # ── Keras interface ───────────────────────────────────────────────────────────

  def get_embedding_table(self):
    return self._embedding_layer.embeddings

  def get_embedding_layer(self):
    return self._embedding_layer

  def get_config(self):
    return dict(self._config)

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)

  @property
  def transformer_layers(self):
    return self._transformer_layers

  @property
  def pooler_layer(self):
    return self._pooler_layer
