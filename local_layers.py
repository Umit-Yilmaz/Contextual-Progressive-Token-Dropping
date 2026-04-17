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

"""Local fallback implementations of token-dropping Keras layers.

TokenImportanceWithMovingAvg and SelectTopK were introduced in
tf-models-official 2.10.x as part of the token dropping project.
This module provides equivalent implementations for use with newer
versions of the package that may not expose them via
official.nlp.modeling.layers.

Usage (handled automatically by encoder.py):
    try:
        layers.TokenImportanceWithMovingAvg  # test presence
        layers.SelectTopK
    except AttributeError:
        layers.TokenImportanceWithMovingAvg = local_layers.TokenImportanceWithMovingAvg
        layers.SelectTopK = local_layers.SelectTopK
"""

import tensorflow as tf
tf_keras = tf.keras  # tf-keras 2.15 standalone incompatible with TF 2.10


class TokenImportanceWithMovingAvg(tf_keras.layers.Layer):
    """Per-vocabulary token importance with exponential moving average updates.

    Maintains a non-trainable float32 embedding table of shape [vocab_size].
    Each entry stores the running-average MLM cross-entropy loss for that
    token type.  Tokens with a higher running-average loss are treated as more
    informative and are prioritised (kept) during token-dropping routing.

    Special token IDs (CLS, SEP, etc.) should be initialised to +1e4 so they
    are always selected, and PAD should be initialised to -1e4 so it is always
    dropped.

    Args:
        vocab_size:          Number of unique token IDs.
        init_importance:     tf.Tensor shape [vocab_size] with initial values.
        moving_average_beta: EMA smoothing factor (higher = slower adaptation).
    """

    def __init__(self, vocab_size, init_importance, moving_average_beta=0.995,
                 **kwargs):
        super().__init__(**kwargs)
        self._vocab_size = vocab_size
        self._init_importance = init_importance
        self._beta = moving_average_beta

    def build(self, input_shape):
        init_val = (
            self._init_importance.numpy()
            if hasattr(self._init_importance, 'numpy')
            else self._init_importance
        )
        self._importance = self.add_weight(
            name='token_importance',
            shape=(self._vocab_size,),
            initializer=tf_keras.initializers.Constant(init_val),
            trainable=False,
            dtype=tf.float32)
        super().build(input_shape)

    def call(self, token_ids):
        """Return importance scores for the given token IDs.

        Args:
            token_ids: int32 tensor of arbitrary shape.
        Returns:
            float32 tensor of the same shape as token_ids.
        """
        return tf.gather(self._importance, token_ids)

    def update_token_importance(self, token_ids, importance):
        """EMA update of per-token importance.

        Called once per training step with the masked token IDs and their
        corresponding per-token MLM cross-entropy losses.

        Args:
            token_ids:  [batch, n_masked] int32 tensor.
            importance: [batch, n_masked] float32 per-token losses.
        """
        flat_ids = tf.cast(tf.reshape(token_ids, [-1]), tf.int32)
        flat_imp = tf.cast(tf.reshape(importance, [-1]), tf.float32)
        current  = tf.gather(self._importance, flat_ids)
        updated  = self._beta * current + (1.0 - self._beta) * flat_imp
        self._importance.assign(
            tf.tensor_scatter_nd_update(
                self._importance,
                tf.expand_dims(flat_ids, axis=1),
                updated))


class SelectTopK(tf_keras.layers.Layer):
    """Splits a sequence into top-k important and remaining tokens.

    Given per-position importance scores, returns:
      • selected:     indices of the top-k most important positions
      • not_selected: indices of the remaining (seq_len − top_k) positions

    Both index tensors are sorted in ascending order to preserve positional
    structure for downstream attention operations.

    Args:
        top_k: Number of tokens to retain in the selected (important) set.
    """

    def __init__(self, top_k, **kwargs):
        super().__init__(**kwargs)
        self._top_k = top_k

    def call(self, importance_scores):
        """Partition token indices by importance.

        Args:
            importance_scores: [batch, seq_len] float32 tensor.
        Returns:
            selected:     [batch, top_k]            int32, ascending
            not_selected: [batch, seq_len - top_k]  int32, ascending
        """
        seq_len = tf.shape(importance_scores)[1]
        # top_k returns (values, indices) sorted DESCENDING — rank all positions
        _, ranked = tf.math.top_k(importance_scores, k=seq_len)
        selected     = tf.sort(ranked[:, :self._top_k], axis=1)
        not_selected = tf.sort(ranked[:, self._top_k:], axis=1)
        return tf.cast(selected, tf.int32), tf.cast(not_selected, tf.int32)
