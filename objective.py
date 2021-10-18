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
"""Contrastive loss functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import tensorflow.compat.v1 as tf

from tensorflow.compiler.tf2xla.python import xla  # pylint: disable=g-direct-tensorflow-import

FLAGS = flags.FLAGS

LARGE_NUM = 1e9


def add_supervised_loss(labels, logits, weights, **kwargs):
  """Compute loss for model and add it to loss collection."""
  return tf.losses.softmax_cross_entropy(labels, logits, weights, **kwargs)


def add_contrastive_loss(hidden,
                         hidden_norm=True,
                         temperature=1.0,
                         tpu_context=None,
                         weights=1.0):
  """Compute the instance discrimination loss for the model.

  Args:
    hidden: hidden vector (`Tensor`) of shape (bsz, dim).
    hidden_norm: whether or not to use normalization on the hidden vector.
    temperature: a `floating` number for temperature scaling.
    tpu_context: context information for tpu.
    weights: a weighting number or vector.

  Returns:
    A loss scalar.
  """
  # Get (normalized) hidden1 and hidden2.
  if hidden_norm:
    hidden = tf.math.l2_normalize(hidden, -1)
  splitted_list = tf.split(hidden, 2 + FLAGS.support_size, 0)
  batch_size = tf.shape(splitted_list[0])[0]

  splitted_list_expanded = [tf.expand_dims(elem, 1) for elem in splitted_list]
  splitted_list_concat = tf.concat(splitted_list_expanded, 1)

  if tpu_context is not None:
    splitted_list_large = [
        tpu_cross_replica_concat(elem, tpu_context) for elem in splitted_list
    ]
    batch_size_large = tf.shape(splitted_list_large[0])[0]
    replica_id = tf.cast(tf.cast(xla.replica_id(), tf.uint32), tf.int32)
    sample_idx = tf.range(batch_size) + replica_id * batch_size
    masks = tf.one_hot(sample_idx, batch_size_large)
  else:
    splitted_list_large = splitted_list
    batch_size_large = batch_size
    masks = tf.one_hot(tf.range(batch_size), batch_size)
  masks_all = tf.tile(masks, [1, len(splitted_list)])

  splitted_list_large_concat = tf.concat(splitted_list_large, 0)

  all_scores = tf.matmul(
      splitted_list_concat, splitted_list_large_concat, transpose_b=True)
  all_scores = tf.reduce_max(all_scores, axis=1)
  all_scores = all_scores - masks_all * LARGE_NUM

  # Topk strategy.
  topk = FLAGS.fn_topk
  _, topk_inds = tf.math.top_k(all_scores, k=topk)

  # Generates mask with elements greater than the threshold.
  threshold = FLAGS.fn_threshold
  dense_mask = tf.math.greater(all_scores, threshold)
  dense_mask = tf.cast(dense_mask, tf.float32)

  # Creates a binary mask with top k elements being 1.0.
  topk_inds_expanded = tf.expand_dims(topk_inds, -1)
  batch_ind = tf.tile(
      tf.reshape(tf.range(batch_size), [-1, 1, 1]), [1, topk, 1])
  taken_ind = tf.reshape(
      tf.concat([batch_ind, topk_inds_expanded], -1), [-1, 2])
  taken_ind = tf.cast(taken_ind, tf.int64)
  sparse_mask = tf.sparse.SparseTensor(
      taken_ind, tf.ones([tf.shape(taken_ind)[0]]),
      [batch_size, tf.shape(splitted_list_large_concat)[0]])
  dense_mask_topk = tf.sparse.to_dense(sparse_mask, validate_indices=False)

  # Combines two masks.
  dense_mask = dense_mask*dense_mask_topk

  # Computes loss.
  total_loss = 0.0
  for anchor_ind in range(len(splitted_list)):
    view_list = []
    log_numerator = 0
    for view_ind in range(len(splitted_list)):
      anchor_view = tf.matmul(
          splitted_list[anchor_ind],
          splitted_list_large[view_ind],
          transpose_b=True) / temperature
      if view_ind == anchor_ind:
        anchor_view = anchor_view - masks*LARGE_NUM
      else:
        log_numerator += tf.reduce_sum(anchor_view * masks, axis=1)
      view_list.append(anchor_view)

    denominator = tf.concat(view_list, 1)
    log_numerator += tf.reduce_sum(denominator * dense_mask, axis=1)
    log_numerator = -log_numerator / (
        len(splitted_list) - 1.0 + tf.reduce_sum(dense_mask, axis=1))
    log_denominator = tf.math.reduce_logsumexp(denominator, axis=1)
    cur_loss = tf.reduce_mean(log_numerator + log_denominator)
    total_loss = total_loss + cur_loss

  total_loss = total_loss*2.0/(len(splitted_list))
  total_loss = total_loss*weights
  tf.losses.add_loss(total_loss)
  return total_loss


def add_contrastive_loss_with_memory(hidden,
                                     memory,
                                     hidden_ema,
                                     hidden_norm=True,
                                     temperature=1.0,
                                     weights=1.0):
  """Compute the instance discrimination loss for the model.

  Args:
    hidden: hidden vector (`Tensor`) of shape (bsz, dim).
    memory: the memory bank.
    hidden_ema: samples from the momentum encoder of the current batch.
    hidden_norm: whether or not to use normalization on the hidden vector.
    temperature: a `floating` number for temperature scaling.
    weights: a weighting number or vector.

  Returns:
    A loss scalar.
  """
  # Get (normalized) hidden1 and hidden2.
  if hidden_norm:
    hidden = tf.math.l2_normalize(hidden, -1)
  splitted_list = tf.split(hidden, 2 + FLAGS.support_size, 0)
  batch_size = tf.shape(splitted_list[0])[0]

  batch_size_large = FLAGS.train_batch_size
  replica_id = tf.cast(tf.cast(xla.replica_id(), tf.uint32), tf.int32)
  sample_idx = tf.range(batch_size) + replica_id * batch_size
  masks = tf.one_hot(sample_idx, batch_size_large)

  masks_all = tf.tile(masks, [1, len(splitted_list)])
  masks_all_mem = tf.concat([
      masks_all,
      tf.zeros([
          batch_size, batch_size_large *
          (FLAGS.memory_multiplier - len(splitted_list))
      ])
  ], -1)

  # Computes the support set from the memory bank.
  splitted_list_mem = tf.split(hidden_ema, 2 + FLAGS.support_size, 0)
  splitted_list_mem_expanded = [
      tf.expand_dims(elem, 1) for elem in splitted_list_mem
  ]
  splitted_list_mem_concat = tf.concat(splitted_list_mem_expanded, 1)

  # Computes similairity scores.
  all_scores = tf.matmul(splitted_list_mem_concat, memory, transpose_b=True)
  all_scores = tf.reduce_max(all_scores, axis=1)  # batch x mem
  all_scores = all_scores - masks_all_mem * LARGE_NUM

  # Topk strategy.
  topk = FLAGS.fn_topk
  _, topk_inds = tf.math.top_k(all_scores, k=topk)

  # Generates mask with elements greater than the threshold.
  threshold = FLAGS.fn_threshold
  dense_mask = tf.math.greater(all_scores, threshold)
  dense_mask = tf.cast(dense_mask, tf.float32)

  # Creates a binary mask with top k elements being 1.0.
  topk_inds_expanded = tf.expand_dims(topk_inds, -1)
  batch_ind = tf.tile(
      tf.reshape(tf.range(batch_size), [-1, 1, 1]), [1, topk, 1])
  taken_ind = tf.reshape(
      tf.concat([batch_ind, topk_inds_expanded], -1), [-1, 2])
  taken_ind = tf.cast(taken_ind, tf.int64)
  sparse_mask = tf.sparse.SparseTensor(
      taken_ind, tf.ones([tf.shape(taken_ind)[0]]),
      [batch_size, tf.shape(memory)[0]])
  dense_mask_topk = tf.sparse.to_dense(sparse_mask, validate_indices=False)

  # Combines two masks.
  dense_mask = dense_mask*dense_mask_topk

  total_loss = 0.0
  for anchor_ind in range(len(splitted_list)):
    # Builds the mask for the current anchor.
    if anchor_ind == 0:
      mask_anchor = tf.concat([
          masks,
          tf.zeros(
              [batch_size, batch_size_large * (FLAGS.memory_multiplier - 1)])
      ], axis=-1)
    else:
      mask_anchor = tf.concat(
          [tf.zeros([batch_size, batch_size_large * anchor_ind]), masks],
          axis=-1)
      mask_anchor = tf.concat([
          mask_anchor,
          tf.zeros([
              batch_size, batch_size_large *
              (FLAGS.memory_multiplier - 1 - anchor_ind)
          ])
      ], axis=-1)

    anchor_mem = tf.matmul(
        splitted_list[anchor_ind], memory, transpose_b=True) / temperature
    log_numerator = tf.reduce_sum(
        anchor_mem * (masks_all_mem * (1. - mask_anchor)), axis=1)
    denominator = anchor_mem - mask_anchor * LARGE_NUM
    log_numerator += tf.reduce_sum(denominator * dense_mask, axis=1)
    log_numerator = -log_numerator / (
        len(splitted_list) - 1.0 + tf.reduce_sum(dense_mask, axis=1))
    log_denominator = tf.math.reduce_logsumexp(denominator, axis=1)
    cur_loss = tf.reduce_mean(log_numerator + log_denominator)
    total_loss = total_loss + cur_loss

  total_loss = total_loss*2.0/(len(splitted_list))
  total_loss = total_loss*weights
  tf.losses.add_loss(total_loss)
  return total_loss


def tpu_cross_replica_concat(tensor, tpu_context=None):
  """Reduce a concatenation of the `tensor` across TPU cores.

  Args:
    tensor: tensor to concatenate.
    tpu_context: A `TPUContext`. If not set, CPU execution is assumed.

  Returns:
    Tensor of the same rank as `tensor` with first dimension `num_replicas`
    times larger.
  """
  if tpu_context is None or tpu_context.num_replicas <= 1:
    return tensor

  num_replicas = tpu_context.num_replicas

  with tf.name_scope('tpu_cross_replica_concat'):
    # This creates a tensor that is like the input tensor but has an added
    # replica dimension as the outermost dimension. On each replica it will
    # contain the local values and zeros for all other values that need to be
    # fetched from other replicas.
    ext_tensor = tf.scatter_nd(
        indices=[[xla.replica_id()]],
        updates=[tensor],
        shape=[num_replicas] + tensor.shape.as_list())

    # As every value is only present on one replica and 0 in all others, adding
    # them all together will result in the full tensor on all replicas.
    ext_tensor = tf.tpu.cross_replica_sum(ext_tensor)

    # Flatten the replica dimension.
    # The first dimension size will be: tensor.shape[0] * num_replicas
    # Using [-1] trick to support also scalar input.
    return tf.reshape(ext_tensor, [-1] + ext_tensor.shape.as_list()[2:])
