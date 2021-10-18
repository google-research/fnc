# coding=utf-8
# Copyright 2020 The SimCLR Authors.
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
# See the License for the specific simclr governing permissions and
# limitations under the License.
# ==============================================================================
"""Model specification for SimCLR."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import data_util as data_util
import model_util as model_util
import objective as obj_lib
import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2

FLAGS = flags.FLAGS


def build_model_fn(model, num_classes, num_train_examples):
  """Build model function."""
  def model_fn(features, labels, mode, params=None):
    """Build model and optimizer."""
    is_training = mode == tf.estimator.ModeKeys.TRAIN

    # Check training mode.
    if FLAGS.train_mode == 'pretrain':
      num_transforms = (2 + FLAGS.support_size)

      if FLAGS.fine_tune_after_block > -1:
        raise ValueError('Does not support layer freezing during pretraining,'
                         'should set fine_tune_after_block<=-1 for safety.')
    elif FLAGS.train_mode == 'finetune':
      num_transforms = 1
    else:
      raise ValueError('Unknown train_mode {}'.format(FLAGS.train_mode))

    batch_size = tf.shape(features)[0]

    # Split channels, and optionally apply extra batched augmentation.
    features_list = tf.split(
        features, num_or_size_splits=num_transforms, axis=-1)
    if FLAGS.use_blur and is_training and FLAGS.train_mode == 'pretrain':
      features_list = data_util.batch_random_blur(
          features_list, FLAGS.image_size, FLAGS.image_size)
    features = tf.concat(features_list, 0)  # (num_transforms * bsz, h, w, c)

    # Base network forward pass.
    with tf.variable_scope('base_model'):
      if FLAGS.train_mode == 'finetune' and FLAGS.fine_tune_after_block >= 4:
        # Finetune just supervised (linear) head will not update BN stats.
        model_train_mode = False
      else:
        # Pretrain or finetune anything else will update BN stats.
        model_train_mode = is_training
      hiddens = model(features, is_training=model_train_mode)

    memory_op = memory = memory_updated = None

    if FLAGS.train_mode == 'pretrain':
      tpu_context = params['context'] if 'context' in params else None

      total_instance = 2+FLAGS.support_size

      hiddens_instance = hiddens[:(total_instance) * batch_size, :]
      hiddens_proj_instance = model_util.projection_head(
          hiddens_instance, is_training, name='head_contrastive_instance')

      if FLAGS.memory_multiplier > 0:
        if FLAGS.memory_multiplier < total_instance:
          raise ValueError(
              'memory_multiplier (%d) needs to be greater or equal to the '
              'number of generated views (%d).' %
              (FLAGS.memory_multiplier, total_instance))
        # Momentum encoder.
        with tf.variable_scope('ema_model', reuse=tf.AUTO_REUSE):
          with tf.variable_scope('base_model', reuse=tf.AUTO_REUSE):
            hiddens_ema = model(features, is_training=model_train_mode)
          hiddens_instance_ema = hiddens_ema[:(total_instance) * batch_size, :]
          hiddens_proj_instance_ema = model_util.projection_head(
              hiddens_instance_ema,
              is_training,
              name='head_contrastive_instance')

        hiddens_proj_instance_ema_singlenode = hiddens_proj_instance_ema
        splitted_list = tf.split(hiddens_proj_instance_ema, total_instance, 0)
        splitted_list_large = [
            obj_lib.tpu_cross_replica_concat(elem, tpu_context)
            for elem in splitted_list
        ]
        hiddens_proj_instance_ema = tf.concat(splitted_list_large, 0)
        if FLAGS.hidden_norm:
          hiddens_proj_instance_ema = tf.math.l2_normalize(
              hiddens_proj_instance_ema, -1)
          hiddens_proj_instance_ema_singlenode = tf.math.l2_normalize(
              hiddens_proj_instance_ema_singlenode, -1)
        hiddens_proj_instance_ema_singlenode = tf.stop_gradient(
            hiddens_proj_instance_ema_singlenode)

        with tf.variable_scope('memory', reuse=tf.AUTO_REUSE):
          memory_real_length = FLAGS.train_batch_size * FLAGS.memory_multiplier
          memory = tf.get_variable(
              'memory',
              [memory_real_length, FLAGS.proj_out_dim],
              trainable=False,
              initializer=tf.zeros_initializer())
          # Updates the memory.
          memory_updated = tf.concat([
              hiddens_proj_instance_ema,
              memory[:-total_instance * FLAGS.train_batch_size]
          ], 0)
          memory_updated = tf.stop_gradient(memory_updated)
          memory_op = tf.assign(memory, memory_updated)

      if memory_updated is None:
        contrast_loss_instance = obj_lib.add_contrastive_loss(
            hiddens_proj_instance,
            hidden_norm=FLAGS.hidden_norm,
            temperature=FLAGS.temperature,
            tpu_context=tpu_context if is_training else None,
            weights=1.0)
      else:
        contrast_loss_instance = obj_lib.add_contrastive_loss_with_memory(
            hiddens_proj_instance,
            memory_updated,
            hiddens_proj_instance_ema_singlenode,
            hidden_norm=FLAGS.hidden_norm,
            temperature=FLAGS.temperature,
            weights=1.0)

      logits_sup = tf.zeros([params['batch_size'], num_classes])
    else:
      contrast_loss_instance = tf.zeros([])
      hiddens = model_util.projection_head(
          hiddens, is_training, name='head_contrastive_instance')
      logits_sup = model_util.supervised_head(
          hiddens, num_classes, is_training)
      obj_lib.add_supervised_loss(
          labels=labels['labels'],
          logits=logits_sup,
          weights=labels['mask'])

    # Adds weight decay to loss, for non-LARS optimizers.
    model_util.add_weight_decay(adjust_per_optimizer=True)
    loss = tf.losses.get_total_loss()

    if FLAGS.train_mode == 'pretrain':
      variables_to_train = [
          var for var in tf.trainable_variables() if (
              not var.name.startswith('ema'))]
    else:
      collection_prefix = 'trainable_variables_inblock_'
      variables_to_train = []
      for j in range(FLAGS.fine_tune_after_block + 1, 6):
        variables_to_train += tf.get_collection(collection_prefix + str(j))
      assert variables_to_train, 'variables_to_train shouldn\'t be empty!'

    tf.logging.info('===============Variables to train (begin)===============')
    tf.logging.info(variables_to_train)
    tf.logging.info('================Variables to train (end)================')

    learning_rate = model_util.learning_rate_schedule(
        FLAGS.learning_rate, num_train_examples)

    if is_training:
      if FLAGS.train_summary_steps > 0:
        # Compute stats for the summary.
        summary_writer = tf2.summary.create_file_writer(FLAGS.model_dir)
        with tf.control_dependencies([summary_writer.init()]):
          with summary_writer.as_default():
            should_record = tf.math.equal(
                tf.math.floormod(tf.train.get_global_step(),
                                 FLAGS.train_summary_steps), 0)
            with tf2.summary.record_if(should_record):
              label_acc = tf.equal(
                  tf.argmax(labels['labels'], 1), tf.argmax(logits_sup, axis=1))
              label_acc = tf.reduce_mean(tf.cast(label_acc, tf.float32))
              tf2.summary.scalar(
                  'train_contrast_loss_instance',
                  contrast_loss_instance,
                  step=tf.train.get_global_step())
              tf2.summary.scalar(
                  'train_label_accuracy',
                  label_acc,
                  step=tf.train.get_global_step())
              tf2.summary.scalar(
                  'learning_rate', learning_rate,
                  step=tf.train.get_global_step())

      optimizer = model_util.get_optimizer(learning_rate)
      control_deps = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      if FLAGS.train_summary_steps > 0:
        control_deps.extend(tf.summary.all_v2_summary_ops())
      with tf.control_dependencies(control_deps):
        train_op = optimizer.minimize(
            loss, global_step=tf.train.get_or_create_global_step(),
            var_list=variables_to_train)
      if (FLAGS.train_mode == 'pretrain') and (FLAGS.memory_multiplier > 0):
        if memory_op is not None:
          train_op = tf.group(train_op, memory_op)

        name2ema_vars = dict(
            [(var.name, var) for var in tf.trainable_variables()])
        ema_mappings = {}  # shadow vars --> real vars
        for name, var in name2ema_vars.items():
          if name.startswith('ema_model'):
            parent_name = name.replace('ema_model/', '')
            ema_mappings[var] = name2ema_vars[parent_name]
        tf.logging.info('===============EMA mapping (begin)===============')
        tf.logging.info(ema_mappings)
        tf.logging.info('================EMA mapping (end)================')

        with tf.control_dependencies([train_op]), tf.name_scope('ema'):
          decay = FLAGS.moving_average_decay
          train_op = []
          for shadow_var, real_var in ema_mappings.items():
            shadow_var_new = decay * shadow_var + (1. - decay) * real_var
            train_op.append(shadow_var.assign(shadow_var_new))
          assert train_op
          train_op = tf.group(train_op)

      if FLAGS.checkpoint:
        def scaffold_fn():
          """Scaffold function to restore non-logits vars from checkpoint."""
          tf.train.init_from_checkpoint(
              FLAGS.checkpoint,
              {v.op.name: v.op.name
               for v in tf.global_variables(FLAGS.variable_schema)})

          if FLAGS.zero_init_logits_layer:
            # Init op that initializes output layer parameters to zeros.
            output_layer_parameters = [
                var for var in tf.trainable_variables() if var.name.startswith(
                    'head_supervised')]
            tf.logging.info('Initializing output layer parameters %s to zero',
                            [x.op.name for x in output_layer_parameters])
            with tf.control_dependencies([tf.global_variables_initializer()]):
              init_op = tf.group([
                  tf.assign(x, tf.zeros_like(x))
                  for x in output_layer_parameters])
            return tf.train.Scaffold(init_op=init_op)
          else:
            return tf.train.Scaffold()
      else:
        scaffold_fn = None

      return tf.estimator.tpu.TPUEstimatorSpec(
          mode=mode, train_op=train_op, loss=loss, scaffold_fn=scaffold_fn)
    else:

      def metric_fn(logits_sup, labels_sup, mask, **kws):
        """Inner metric function."""
        metrics = {k: tf.metrics.mean(v, weights=mask)
                   for k, v in kws.items()}
        metrics['label_top_1_accuracy'] = tf.metrics.accuracy(
            tf.argmax(labels_sup, 1), tf.argmax(logits_sup, axis=1),
            weights=mask)
        metrics['label_top_5_accuracy'] = tf.metrics.recall_at_k(
            tf.argmax(labels_sup, 1), logits_sup, k=5, weights=mask)
        return metrics

      metrics = {
          'logits_sup':
              logits_sup,
          'labels_sup':
              labels['labels'],
          'mask':
              labels['mask'],
          'contrast_loss':
              tf.fill((params['batch_size'],), contrast_loss_instance),
          'regularization_loss':
              tf.fill((params['batch_size'],),
                      tf.losses.get_regularization_loss()),
      }

      return tf.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=loss,
          eval_metrics=(metric_fn, metrics),
          scaffold_fn=None)

  return model_fn
