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

"""Config and factory for the Progressive Contextual Drop encoder."""

import dataclasses
import os
import sys
from typing import Tuple

import tensorflow as tf
tf_keras = tf.keras

from official.modeling import tf_utils
from official.modeling.hyperparams import base_config
from official.nlp.configs import encoders

# Make the encoder module importable regardless of how this package is invoked.
sys.path.insert(0, os.path.dirname(__file__))
from encoder import ProgressiveContextualDropEncoder  # noqa: E402


@dataclasses.dataclass
class ProgressiveDropEncoderConfig(encoders.BertEncoderConfig):
  """Configuration for ProgressiveContextualDropEncoder.

  Inherits all standard BertEncoderConfig fields and adds three
  progressive-dropping budget parameters plus allow/deny lists.

  Fields (in addition to BertEncoderConfig):
    token_keep_k1: Tokens kept after drop stage 1.  Default 384 (75 % of 512).
    token_keep_k2: Tokens kept after drop stage 2.  Default 256 (50 % of 512).
    token_keep_k3: Tokens kept after drop stage 3.  Default 128 (25 % of 512).
    token_allow_list: Token IDs never dropped  (special tokens).
    token_deny_list:  Token IDs always dropped (padding).
  """
  token_keep_k1: int = 384
  token_keep_k2: int = 256
  token_keep_k3: int = 128
  token_allow_list: Tuple[int, ...] = (100, 101, 102, 103)
  token_deny_list: Tuple[int, ...]  = (0,)


@base_config.bind(ProgressiveDropEncoderConfig)
def get_encoder(encoder_cfg: ProgressiveDropEncoderConfig
                ) -> ProgressiveContextualDropEncoder:
  """Instantiate a ProgressiveContextualDropEncoder from config."""
  return ProgressiveContextualDropEncoder(
      vocab_size=encoder_cfg.vocab_size,
      hidden_size=encoder_cfg.hidden_size,
      num_layers=encoder_cfg.num_layers,
      num_attention_heads=encoder_cfg.num_attention_heads,
      inner_dim=encoder_cfg.intermediate_size,
      inner_activation=tf_utils.get_activation(encoder_cfg.hidden_activation),
      output_dropout=encoder_cfg.dropout_rate,
      attention_dropout=encoder_cfg.attention_dropout_rate,
      max_sequence_length=encoder_cfg.max_position_embeddings,
      type_vocab_size=encoder_cfg.type_vocab_size,
      initializer=tf_keras.initializers.TruncatedNormal(
          stddev=encoder_cfg.initializer_range),
      output_range=encoder_cfg.output_range,
      embedding_width=encoder_cfg.embedding_size,
      norm_first=encoder_cfg.norm_first,
      token_keep_k1=encoder_cfg.token_keep_k1,
      token_keep_k2=encoder_cfg.token_keep_k2,
      token_keep_k3=encoder_cfg.token_keep_k3,
      token_allow_list=encoder_cfg.token_allow_list,
      token_deny_list=encoder_cfg.token_deny_list)
