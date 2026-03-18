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

"""Progressive Contextual Drop BERT experiment configuration.

Registers the experiment name 'progressive_drop_bert/pretraining' so that
train.py can be invoked with:

  python train.py \\
    --experiment=progressive_drop_bert/pretraining \\
    --config_file=<path>/wiki_books_pretrain_sequence_pack.yaml \\
    --config_file=<path>/bert_progressive_drop.yaml \\
    ...

The resulting checkpoint has the same structure as standard BERT and can be
used directly for fine-tuning with the regular BERT pipeline.
"""

import os
import sys

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import optimization
from official.nlp.configs import bert
from official.nlp.configs import encoders
from official.nlp.data import pretrain_dataloader

# Ensure this package's modules are importable.
sys.path.insert(0, os.path.dirname(__file__))
from encoder_config import ProgressiveDropEncoderConfig  # noqa: E402
from masked_lm import ProgressiveDropMaskedLMConfig       # noqa: E402


@exp_factory.register_config_factory('progressive_drop_bert/pretraining')
def progressive_drop_bert_pretraining() -> cfg.ExperimentConfig:
  """BERT pretraining with Progressive Contextual Token Dropping."""
  return cfg.ExperimentConfig(
      runtime=cfg.RuntimeConfig(enable_xla=True),
      task=ProgressiveDropMaskedLMConfig(
          model=bert.PretrainerConfig(
              encoder=encoders.EncoderConfig(
                  any=ProgressiveDropEncoderConfig(
                      vocab_size=30522,
                      num_layers=12,
                      token_keep_k1=384,
                      token_keep_k2=256,
                      token_keep_k3=128),
                  type='any')),
          train_data=pretrain_dataloader.BertPretrainDataConfig(),
          validation_data=pretrain_dataloader.BertPretrainDataConfig(
              is_training=False)),
      trainer=cfg.TrainerConfig(
          train_steps=1_000_000,
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'adamw',
                  'adamw': {
                      'weight_decay_rate': 0.01,
                      'exclude_from_weight_decay': [
                          'LayerNorm', 'layer_norm', 'bias'],
                  },
              },
              'learning_rate': {
                  'type': 'polynomial',
                  'polynomial': {
                      'initial_learning_rate': 1e-4,
                      'end_learning_rate': 0.0,
                  },
              },
              'warmup': {'type': 'polynomial'},
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None',
      ])
