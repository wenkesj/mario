from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


def normalized_columns_initializer(stdv=1.0):
  """factory for normalizing over columns given stdv."""
  def initializer(shape, dtype=None, partition_info=None):
    out = np.random.randn(*shape).astype(np.float32)
    out *= stdv / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
    return tf.constant(out)
  return initializer


def flatten(x):
  """flatten to a 1d tensor."""
  return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])


def conv2d(x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="SAME", dtype=tf.float32, collections=None):
  """conv2d wrapper."""
  with tf.variable_scope(name):
    stride_shape = [1, stride[0], stride[1], 1]
    filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]), num_filters]
    fan_in = np.prod(filter_shape[:3])
    fan_out = np.prod(filter_shape[:2]) * num_filters
    w_bound = np.sqrt(6. / (fan_in + fan_out))

    w = tf.get_variable("w", filter_shape, dtype, tf.random_uniform_initializer(-w_bound, w_bound),
                        collections=collections)
    b = tf.get_variable("b", [1, 1, 1, num_filters], initializer=tf.constant_initializer(0.0),
                        collections=collections)
    return tf.nn.conv2d(x, w, stride_shape, pad) + b


def linear(x, size, name, initializer=None, bias_init=0):
  """linear layer"""
  with tf.variable_scope(name):
    w = tf.get_variable("w", [x.get_shape()[1], size], initializer=initializer)
    b = tf.get_variable("b", [size], initializer=tf.constant_initializer(bias_init))
    y = tf.matmul(x, w) + b
  return y


def lstm(x, size, actions, apply_softmax=False):
  """Simple LSTM implementation.

  Here we roll out over the batch so it makes it logically easier.
  input: (1, batch_size, n_input)
  """
  x = tf.expand_dims(x, [0])
  lstm = tf.contrib.rnn.BasicLSTMCell(size, state_is_tuple=True)
  state_size = lstm.state_size
  step_size = tf.shape(x)[:1]
  c_init = np.zeros((1, state_size.c), np.float32)
  h_init = np.zeros((1, state_size.h), np.float32)
  state_init = [c_init, h_init]
  c_in = tf.placeholder(tf.float32, [1, state_size.c], name='c_in')
  h_in = tf.placeholder(tf.float32, [1, state_size.h], name='h_in')
  state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
  lstm_outputs, lstm_state = tf.nn.dynamic_rnn(lstm, x,
                                               initial_state=state_in,
                                               sequence_length=step_size,
                                               time_major=False)
  lstm_c, lstm_h = lstm_state
  x = tf.reshape(lstm_outputs, [-1, size])
  state_out = [lstm_c[:1, :], lstm_h[:1, :]]
  y = linear(x, actions, "logits", normalized_columns_initializer(0.01))
  if apply_softmax:
    y = tf.nn.softmax(y, dim=-1)
  return y, state_in, state_out, state_init


def batch_norm_conv2d(x, num_filters, name, filter_size=(3, 3),
                      stride=(1, 1), act=tf.nn.relu, pad="SAME",
                      dtype=tf.float32, collections=None):
  return act(tf.layers.batch_normalization(conv2d(x, num_filters, name, filter_size, stride)))


def mario_net(name, use_lstm=False, lstm_size=256, actions=13, apply_softmax=False):
  """This is inspired by "Learning by Prediction", added an LSTM.

  input: (None, width, height, channels)
  output: (None, 1280) -> (None, 512) -> (None, actions)
  """
  def net(x, *args, **kwargs):
    x = x['state']
    x = batch_norm_conv2d(x, 32, "l1", [5, 5], [4, 4], tf.nn.relu)
    x = batch_norm_conv2d(x, 32, "l2", [3, 3], [2, 2], tf.nn.relu)
    x = batch_norm_conv2d(x, 32, "l3", [3, 3], [2, 2], tf.nn.relu)
    x = batch_norm_conv2d(x, 64, "l4", [3, 3], [2, 2], tf.nn.relu)
    x = flatten(x)
    x = tf.nn.relu(linear(x, 256, "fc", normalized_columns_initializer(0.01)))
    with tf.variable_scope(name):
      if use_lstm:
        return lstm(x, lstm_size, actions, apply_softmax)
      logits = linear(x, actions, "logits", normalized_columns_initializer(0.01))
      return tf.nn.softmax(logits, dim=-1)
  return net
