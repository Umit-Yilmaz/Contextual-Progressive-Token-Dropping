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

"""Vanilla BERT pretraining experiment — 3-way comparison baseline.

Registers the experiment name 'vanilla_bert/pretraining' using the standard
BertEncoderV2 (no token dropping whatsoever) as the reference baseline for
comparing TokenDrop and Progressive Drop methods.

Metrics tracked (all three experiments share the same set for fair comparison):
  - lm_example_loss    : mean MLM cross-entropy per masked token
  - masked_lm_accuracy : fraction of masked tokens correctly predicted
  - next_sentence_loss : NSP loss if the data provides NSP labels

Import this module in train.py alongside experiment_configs.py and the
progressive experiment_configs.py so all three experiments are registered.
"""

import dataclasses
import os
import sys

import tensorflow as tf
tf_keras = tf.keras  # tf-keras 2.15 standalone incompatible with TF 2.10

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.core import task_factory
from official.modeling import optimization
from official.nlp.configs import bert
from official.nlp.configs import encoders
from official.nlp.data import pretrain_dataloader
from official.nlp.tasks import masked_lm as official_masked_lm


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class VanillaBertMaskedLMConfig(official_masked_lm.MaskedLMConfig):
    """Config for standard (no-drop) BERT MLM task.  No extra fields needed."""
    pass


# ---------------------------------------------------------------------------
# Task — adds masked_lm_accuracy metric on top of the official MaskedLMTask
# ---------------------------------------------------------------------------

@task_factory.register_task_cls(VanillaBertMaskedLMConfig)
class VanillaBertMaskedLMTask(official_masked_lm.MaskedLMTask):
    """Standard BERT MLM pretraining task with accuracy tracking.

    Identical to the official MaskedLMTask but additionally records
    masked_lm_accuracy so that all three competing models (vanilla,
    token-drop, progressive-drop) can be compared on equal footing.
    """

    def build_metrics(self, training=None):
        metrics = super().build_metrics(training=training)
        metrics.append(tf_keras.metrics.Mean(name='masked_lm_accuracy'))
        return metrics

    def build_losses(self, labels, model_outputs, metrics, aux_losses=None):
        """Compute MLM (+ optional NSP) loss and update all metrics.

        Args:
            labels:        Dict with 'masked_lm_ids', 'masked_lm_weights',
                           and optionally 'next_sentence_labels'.
            model_outputs: Dict with 'mlm_logits' and optionally 'next_sentence'.
            metrics:       List of tf.keras.metrics.Metric objects.
            aux_losses:    Optional list of regularisation losses.

        Returns:
            total_loss: scalar tf.Tensor.
        """
        with tf.name_scope('VanillaBertMaskedLMTask/losses'):
            metrics_dict = {m.name: m for m in metrics}

            # ── MLM loss ──────────────────────────────────────────────────
            lm_logits = tf.cast(model_outputs['mlm_logits'], tf.float32)
            lm_per_token = tf_keras.losses.sparse_categorical_crossentropy(
                labels['masked_lm_ids'], lm_logits, from_logits=True)
            weights = labels['masked_lm_weights']
            denom   = tf.reduce_sum(weights)
            mlm_loss = tf.math.divide_no_nan(
                tf.reduce_sum(lm_per_token * weights), denom)
            metrics_dict['lm_example_loss'].update_state(mlm_loss)

            # ── MLM accuracy ──────────────────────────────────────────────
            if 'masked_lm_accuracy' in metrics_dict:
                preds = tf.argmax(lm_logits, axis=-1, output_type=tf.int32)
                per_tok_acc = tf.cast(
                    tf.equal(preds, labels['masked_lm_ids']), tf.float32
                ) * weights
                mlm_acc = tf.math.divide_no_nan(
                    tf.reduce_sum(per_tok_acc), denom)
                metrics_dict['masked_lm_accuracy'].update_state(mlm_acc)

            # ── NSP loss (optional) ───────────────────────────────────────
            if 'next_sentence_labels' in labels:
                nsp_loss = tf.reduce_mean(
                    tf_keras.losses.sparse_categorical_crossentropy(
                        labels['next_sentence_labels'],
                        tf.cast(model_outputs['next_sentence'], tf.float32),
                        from_logits=True))
                if 'next_sentence_loss' in metrics_dict:
                    metrics_dict['next_sentence_loss'].update_state(nsp_loss)
                total_loss = mlm_loss + nsp_loss
            else:
                total_loss = mlm_loss

            if aux_losses:
                total_loss += tf.add_n(aux_losses)

            return total_loss

    def train_step(self, inputs, model: tf_keras.Model,
                   optimizer: tf_keras.optimizers.Optimizer, metrics):
        """Forward + backward pass."""
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
        grads = tape.gradient(
            scaled_loss if self.task_config.scale_loss else loss, tvars)
        optimizer.apply_gradients(list(zip(grads, tvars)))
        self.process_metrics(metrics, inputs, outputs)
        return {self.loss: loss}

    def validation_step(self, inputs, model: tf_keras.Model, metrics):
        """Validation forward pass."""
        outputs = self.inference_step(inputs, model)
        loss    = self.build_losses(
            labels=inputs,
            model_outputs=outputs,
            metrics=metrics,
            aux_losses=model.losses)
        self.process_metrics(metrics, inputs, outputs)
        return {self.loss: loss}


# ---------------------------------------------------------------------------
# Experiment factory
# ---------------------------------------------------------------------------

@exp_factory.register_config_factory('vanilla_bert/pretraining')
def vanilla_bert_pretraining() -> cfg.ExperimentConfig:
    """Standard BERT-base pretraining without any token dropping.

    Serves as the reference baseline for the 3-way comparison:
        vanilla_bert        ← this experiment
        token_drop_bert     ← experiment_configs.py
        progressive_drop_bert ← experiments/progressive_contextual_dropping/
    """
    return cfg.ExperimentConfig(
        runtime=cfg.RuntimeConfig(enable_xla=True),
        task=VanillaBertMaskedLMConfig(
            model=bert.PretrainerConfig(
                encoder=encoders.EncoderConfig(
                    # Use the built-in BERT encoder (BertEncoderV2) — no dropping.
                    # BERT-base defaults: hidden=768, layers=12, heads=12.
                    # Field values can be overridden via --params_override or
                    # a bert_en_uncased_base_vanilla.yaml config file.
                    bert=encoders.BertEncoderConfig(
                        vocab_size=30522,
                        hidden_size=768,
                        num_layers=12,
                        num_attention_heads=12,
                        max_sequence_length=512,
                        type_vocab_size=2,
                        intermediate_size=3072,
                        inner_activation='gelu',
                        output_dropout=0.1,
                        attention_dropout=0.1,
                        initializer_range=0.02,
                    ),
                    type='bert')),
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
