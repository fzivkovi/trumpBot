# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""
MOTIVATION:
Quoting from Pointer Sentinel Mixtrue Model (https://arxiv.org/abs/1609.07843), section 5.1:
"K1=K2 ... This results in only a single RNN output receiving backpropagation for k timesteps, 
with the other extreme being that the first token receives backpropagation for 0 timesteps."

TRADEOFFS:
The smaller K1 is, the more you utilize the data. 
The bigger K1 is, the faster the training. Less you utilize the data.

COMMENTS, K1 = 1:
In Merity's paper, that is why K1 = 1.
When the decoder is just a classifier, it makes sense to use data more efficiently.
COMMENTS, K1 != 1:
In (Zaremba, et. al.), K1 = K2. 
For Seq2Seq, K1 is decoder length. We cannot make K1=1.

WHAT HAVE I DONE:
I made K1 and K2 configurable.

HOW: 
We can generalize this code to be CONFIGURABLE for both (pseudocode):
Grab K1 data
if K1 < K2:
  prepend K2-K1 extra elements.
if K1 > K2: 
  return Error (doesnt make sense to support)

NOTATION:
K2 - K1 --> L
K1 --> num_steps

"""

import collections
import os
import sys
import tensorflow as tf


def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    return f.read().decode("utf-8").replace("\n", "<eos>").split()


def _build_vocab(filename):
  data = _read_words(filename)

  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))

  return word_to_id


def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  return [word_to_id[word] for word in data if word in word_to_id]


def ptb_raw_data(data_path=None):
  """Load PTB raw data from data directory "data_path".

  Reads PTB text files, converts strings to integer ids,
  and performs mini-batching of the inputs.

  The PTB dataset comes from Tomas Mikolov's webpage:

  http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

  Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.

  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
  """

  train_path = os.path.join(data_path, "ptb.train.txt")
  valid_path = os.path.join(data_path, "ptb.valid.txt")
  test_path = os.path.join(data_path, "ptb.test.txt")

  word_to_id = _build_vocab(train_path)
  train_data = _file_to_word_ids(train_path, word_to_id)
  valid_data = _file_to_word_ids(valid_path, word_to_id)
  test_data = _file_to_word_ids(test_path, word_to_id)
  vocabulary = len(word_to_id)
  return train_data, valid_data, test_data, vocabulary, word_to_id


def ptb_producer(raw_data, batch_size, num_steps, L, name=None):

  """Iterate on the raw PTB data.

  This chunks up raw_data into batches of examples and returns Tensors that
  are drawn from these batches.

  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.
    L: extra data on input for context.
    name: the name of this operation (optional).

  Returns:
    A pair of Tensors, each shaped [batch_size, num_steps]. The second element
    of the tuple is the same data time-shifted to the right by one.

  Raises:
    tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
  """
  with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

    data_len = tf.size(raw_data)
    batch_len = data_len // batch_size
    data = tf.reshape(raw_data[0 : batch_size * batch_len],
                      [batch_size, batch_len])

    epoch_size = (batch_len - 1) // num_steps
    assertion = tf.assert_positive(
        epoch_size,
        message="epoch_size == 0, decrease batch_size or num_steps")
    with tf.control_dependencies([assertion]):
      epoch_size = tf.identity(epoch_size, name="epoch_size")

    # Can't index further back than available.
    throw_away = (L // num_steps) + 1

    i = tf.train.range_input_producer(epoch_size-throw_away, shuffle=False).dequeue()

    x = tf.strided_slice(data, [0, (i+throw_away) * num_steps-L],
                         [batch_size, ((i+throw_away) + 1) * num_steps])
    x.set_shape([batch_size, num_steps+L])


    y = tf.strided_slice(data, [0, (i+throw_away) * num_steps + 1],
                         [batch_size, ((i+throw_away) + 1) * num_steps + 1])
    y.set_shape([batch_size, num_steps])

    # x = tf.Print(x, [x, tf.shape(x)],message="x: ", summarize=100)
    # y = tf.Print(y, [y, tf.shape(y)],message="y: ", summarize=100)

    return x, y


