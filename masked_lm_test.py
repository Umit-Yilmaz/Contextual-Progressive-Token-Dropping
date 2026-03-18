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

"""Tests for official.nlp.tasks.masked_lm."""

import os
import sys

import tensorflow as tf
tf_keras = tf.keras

from official.nlp.configs import bert
from official.nlp.configs import encoders
from official.nlp.data import pretrain_dataloader

# Production fix: replace broken official.projects.token_dropping imports
# with local modules.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from encoder_config import TokenDropBertEncoderConfig   # noqa: E402
from masked_lm import TokenDropMaskedLMConfig, TokenDropMaskedLMTask  # noqa: E402


class MLMTaskTest(tf.test.TestCase):

  def test_task(self):
    config = TokenDropMaskedLMConfig(
        init_checkpoint=self.get_temp_dir(),
        scale_loss=True,
        model=bert.PretrainerConfig(
            encoder=encoders.EncoderConfig(
                any=TokenDropBertEncoderConfig(
                    vocab_size=30522, num_layers=1, token_keep_k=64),
                type="any"),
            cls_heads=[
                bert.ClsHeadConfig(
                    inner_dim=10, num_classes=2, name="next_sentence")
            ]),
        train_data=pretrain_dataloader.BertPretrainDataConfig(
            input_path="dummy",
            max_predictions_per_seq=20,
            seq_length=128,
            global_batch_size=1))
    task = TokenDropMaskedLMTask(config)
    model = task.build_model()
    metrics = task.build_metrics()
    dataset = task.build_inputs(config.train_data)

    iterator = iter(dataset)
    optimizer = tf_keras.optimizers.SGD(lr=0.1)
    task.train_step(next(iterator), model, optimizer, metrics=metrics)
    task.validation_step(next(iterator), model, metrics=metrics)

    # Saves a checkpoint.
    ckpt = tf.train.Checkpoint(model=model, **model.checkpoint_items)
    ckpt.save(config.init_checkpoint)
    task.initialize(model)


if __name__ == "__main__":
  tf.test.main()
