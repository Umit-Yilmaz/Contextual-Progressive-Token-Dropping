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

"""Masked-LM pretraining task for Progressive Contextual Drop BERT.

Key difference from the baseline TokenDropMaskedLMTask:
  - Does NOT call encoder_network.record_mlm_loss() because the progressive
    encoder derives routing decisions from live hidden-state norms, not from a
    historical vocabulary-level MLM-loss table.
  - build_losses returns only the scalar total_loss (no per-token tuple),
    keeping train_step simple.
"""

import dataclasses

import tensorflow as tf
tf_keras = tf.keras

from official.core import task_factory
from official.nlp.tasks import masked_lm


@dataclasses.dataclass
class ProgressiveDropMaskedLMConfig(masked_lm.MaskedLMConfig):
  """Config for progressive drop masked-LM task (no extra fields needed)."""
  pass


@task_factory.register_task_cls(ProgressiveDropMaskedLMConfig)
class ProgressiveDropMaskedLMTask(masked_lm.MaskedLMTask):
  """MLM pretraining task for ProgressiveContextualDropEncoder.

  Routing is performed inside the encoder at forward-pass time using the
  current hidden states, so no post-step bookkeeping is required here.
  """

  def build_metrics(self, training=None):
    """Extend parent metrics with masked-token prediction accuracy."""
    metrics = super().build_metrics(training=training)
    metrics.append(tf_keras.metrics.Mean(name='masked_lm_accuracy'))
    return metrics

  def build_losses(self, labels, model_outputs, metrics,
                   aux_losses=None) -> tf.Tensor:
    """Compute total pretraining loss.

    Args:
      labels:       Dict of label tensors (masked_lm_ids, masked_lm_weights,
                    optionally next_sentence_labels).
      model_outputs: Dict of model output tensors (mlm_logits, optionally
                    next_sentence).
      metrics:      List of Keras metric objects.
      aux_losses:   Optional list of regularisation losses from model.losses.

    Returns:
      total_loss: Scalar tf.Tensor.
    """
    with tf.name_scope('ProgressiveDropMaskedLMTask/losses'):
      metrics_dict = {m.name: m for m in metrics}

      lm_per_token = tf_keras.losses.sparse_categorical_crossentropy(
          labels['masked_lm_ids'],
          tf.cast(model_outputs['mlm_logits'], tf.float32),
          from_logits=True)
      weights      = labels['masked_lm_weights']
      mlm_loss     = tf.math.divide_no_nan(
          tf.reduce_sum(lm_per_token * weights),
          tf.reduce_sum(weights))
      metrics_dict['lm_example_loss'].update_state(mlm_loss)
      if 'masked_lm_accuracy' in metrics_dict:
        prog_preds = tf.argmax(
            tf.cast(model_outputs['mlm_logits'], tf.float32),
            axis=-1, output_type=tf.int32)
        denom = tf.reduce_sum(weights)
        per_tok_acc = tf.cast(
            tf.equal(prog_preds, labels['masked_lm_ids']), tf.float32
        ) * weights
        prog_acc = tf.math.divide_no_nan(tf.reduce_sum(per_tok_acc), denom)
        metrics_dict['masked_lm_accuracy'].update_state(prog_acc)

      if 'next_sentence_labels' in labels:
        nsp_loss = tf.reduce_mean(
            tf_keras.losses.sparse_categorical_crossentropy(
                labels['next_sentence_labels'],
                tf.cast(model_outputs['next_sentence'], tf.float32),
                from_logits=True))
        metrics_dict['next_sentence_loss'].update_state(nsp_loss)
        total_loss = mlm_loss + nsp_loss
      else:
        total_loss = mlm_loss

      if aux_losses:
        total_loss += tf.add_n(aux_losses)

      return total_loss

  def train_step(self, inputs, model: tf_keras.Model,
                 optimizer: tf_keras.optimizers.Optimizer, metrics):
    """Forward + backward pass.  No record_mlm_loss call needed."""
    with tf.GradientTape() as tape:
      outputs = model(inputs, training=True)
      loss    = self.build_losses(
          labels=inputs,
          model_outputs=outputs,
          metrics=metrics,
          aux_losses=model.losses)
      if self.task_config.scale_loss:
        scaled_loss = loss / tf.distribute.get_strategy().num_replicas_in_sync

    tvars = model.trainable_variables
    if self.task_config.scale_loss:
      grads = tape.gradient(scaled_loss, tvars)
    else:
      grads = tape.gradient(loss, tvars)
    optimizer.apply_gradients(list(zip(grads, tvars)))
    self.process_metrics(metrics, inputs, outputs)
    return {self.loss: loss}

  def validation_step(self, inputs, model: tf_keras.Model, metrics):
    outputs = self.inference_step(inputs, model)
    loss = self.build_losses(
        labels=inputs,
        model_outputs=outputs,
        metrics=metrics,
        aux_losses=model.losses)
    self.process_metrics(metrics, inputs, outputs)
    return {self.loss: loss}
