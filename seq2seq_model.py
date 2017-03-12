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

"""Sequence-to-sequence model with an attention mechanism."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.models.rnn.translate import data_utils

import sys
import os

import config

from six.moves import zip   

from tensorflow.python import shape
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope

from tensorflow.python.ops import seq2seq as seq2seqFile

from tensorflow.python.util import nest

# TODO(ebrevdo): Remove once _linear is fully deprecated.
linear = rnn_cell._linear  # pylint: disable=protected-access

from localWrapper import LocalEmbeddingWrapper, load_embedding


def local_extract_argmax_and_embed(embedding, output_projection=None,
                              update_embedding=True):


  # print("HEY FILIP YAY ACTUALLY USING THIS...NOW DELETE THE PRINT.")

  """Get a loop_function that extracts the previous symbol and embeds it.

  Args:
    embedding: embedding tensor for symbols.
    output_projection: None or a pair (W, B). If provided, each fed previous
      output will first be multiplied by W and added B.
    update_embedding: Boolean; if False, the gradients will not propagate
      through the embeddings.

  Returns:
    A loop function.
  """
  def loop_function(prev, _):
    if output_projection is not None:
      prev = nn_ops.xw_plus_b(
          prev, output_projection[0], output_projection[1])
    prev_symbol = math_ops.argmax(prev, 1)
    # Note that gradients will not propagate through the second parameter of
    # embedding_lookup.
    emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
    if not update_embedding:
      emb_prev = array_ops.stop_gradient(emb_prev)
    return emb_prev
  return loop_function




def local_attention_decoder(decoder_inputs,
                      initial_state,
                      attention_states,
                      cell,
                      output_size=None,
                      num_heads=1,
                      loop_function=None,
                      dtype=None,
                      scope=None,
                      initial_state_attention=False):


  if not decoder_inputs:
    raise ValueError("Must provide at least 1 input to attention decoder.")
  if num_heads < 1:
    raise ValueError("With less than 1 heads, use a non-attention decoder.")
  if attention_states.get_shape()[2].value is None:
    raise ValueError("Shape[2] of attention_states must be known: %s"
                     % attention_states.get_shape())
  if output_size is None:
    output_size = cell.output_size

  with variable_scope.variable_scope(
      scope or "attention_decoder", dtype=dtype) as scope:
    dtype = scope.dtype

    batch_size = array_ops.shape(decoder_inputs[0])[0]  # Needed for reshaping.
    attn_length = attention_states.get_shape()[1].value
    if attn_length is None:
      attn_length = shape(attention_states)[1]

    attn_size = attention_states.get_shape()[2].value

    # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
    hidden = array_ops.reshape(
        attention_states, [-1, attn_length, 1, attn_size])
    hidden_features = []
    v = []

    ###### OUR CODE HERE.
    ## This was needed for the extension methods.
    attention_vec_size = attn_size 
    if config.attention_type != 'vinyals':
      attention_vec_size = attn_size * config.num_layers # Size of query vectors for attention.
    ###### END OUR CODE.

    for a in xrange(num_heads):
      ## FFZZZZ ## THIS IS WHERE W COMES FROM.
      k = variable_scope.get_variable("AttnW_%d" % a,
                                      [1, 1, attn_size, attention_vec_size],initializer=tf.contrib.layers.xavier_initializer())
      hidden_features.append(nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
      v.append(
          variable_scope.get_variable("AttnV_%d" % a, [attention_vec_size], initializer=tf.contrib.layers.xavier_initializer()))

    state = initial_state

    def attention(query):
      """Put attention masks on hidden using hidden_features and query."""
      ds = []  # Results of attention reads will be stored here.
      if nest.is_sequence(query):  # If the query is a tuple, flatten it.
        query_list = nest.flatten(query)
        for q in query_list:  # Check that ndims == 2 if specified.
          ndims = q.get_shape().ndims
          if ndims:
            assert ndims == 2
        query = array_ops.concat(1, query_list)
      # print('num_heads',num_heads)
      for a in xrange(num_heads):
        with variable_scope.variable_scope("Attention_%d" % a):

          ###### OUR CODE HERE.
          # DIAGNOSIS
          # attention_vec_size --> 64
          # Hidden features: shape=(?, 20, 1, 64), batchsize X EncoderBucketSize x numHeads x hiddenStatesDim
          # query: shape=(?, 64) hiddenStatesDim
          # print('attention_vec_size ', attention_vec_size) 
          # print('query ', query)
          # print('hidden_features ', hidden_features)
          # print('hidden_features[a]', hidden_features[a])
          if config.attention_type == 'vinyals':
            y = linear(query, attention_vec_size, True)
            y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
            # Attention mask is a softmax of v^T * tanh(...).
            s = math_ops.reduce_sum(
                v[a] * math_ops.tanh(hidden_features[a] + y), [2, 3])
          elif config.attention_type == "luong":
            ## IMPLEMENTATION:
            # Rather than this: new_attn = softmax(V^T * tanh(W * attention_states + U * new_state))
            # We want this: new_attn = softmax( query * W * attention_states )
            # hidden_features = (W * attention_states)
            # Therefore:
            # print("\n")
            # print('query: ', query)
            # print('attention_vec_size: ', attention_vec_size)
            # print('hidden_features[a]: ',hidden_features[a])
            y = array_ops.reshape(query, [-1, 1, 1, attention_vec_size])
            # print('y: ', y)
            # print("\n")
            s = math_ops.reduce_sum(hidden_features[a] * y, [2, 3])
          elif config.attention_type == "bahdanau":
            y = array_ops.reshape(query, [-1, 1, 1, attention_vec_size])
            s = math_ops.reduce_sum(hidden * y, [2, 3])
          else:
            print("...pick attention type.")
            sys.exit()

          a_soft = nn_ops.softmax(s)
          if config.mode == 'test':
            a_soft = tf.Print(a_soft, [a_soft], message="where I'm paying attention: ", first_n=100, summarize=200)
          ###### END OUR CODE

          # Now calculate the attention-weighted vector d.
          d = math_ops.reduce_sum(
              array_ops.reshape(a_soft, [-1, attn_length, 1, 1]) * hidden,
              [1, 2])
          ds.append(array_ops.reshape(d, [-1, attn_size]))

      # print("Final 'H' that gets appended.")
      # print(ds)
      # ds[0] = tf.Print(ds[0], [ds[0]], message=".... ", first_n=100, summarize=200)
      return ds

    outputs = []
    prev = None
    batch_attn_size = array_ops.pack([batch_size, attn_size])
    attns = [array_ops.zeros(batch_attn_size, dtype=dtype)
             for _ in xrange(num_heads)]
    for a in attns:  # Ensure the second shape of attention vectors is set.
      a.set_shape([None, attn_size])

    if initial_state_attention:
      attns = attention(initial_state)

    for i, inp in enumerate(decoder_inputs):
      if i > 0:
        variable_scope.get_variable_scope().reuse_variables()
      # If loop_function is set, we use it instead of decoder_inputs.
      if loop_function is not None and prev is not None:
        with variable_scope.variable_scope("loop_function", reuse=True):
          inp = loop_function(prev, i)
      # Merge input and previous attentions into one vector of the right size.
      input_size = inp.get_shape().with_rank(2)[1]
      if input_size.value is None:
        raise ValueError("Could not infer input size from input: %s" % inp.name)
      x = linear([inp] + attns, input_size, True)
      # Run the RNN.
      cell_output, state = cell(x, state)
      # Run the attention mechanism.
      if i == 0 and initial_state_attention:
        with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                           reuse=True):
          attns = attention(state)


      else:
        attns = attention(state)

      with variable_scope.variable_scope("AttnOutputProjection"):
        output = linear([cell_output] + attns, output_size, True)
      if loop_function is not None:
        prev = output
      outputs.append(output)

  return outputs, state



def local_embedding_attention_decoder(decoder_inputs,
                                initial_state,
                                attention_states,
                                cell,
                                num_symbols,
                                embedding_size,
                                num_heads=1,
                                output_size=None,
                                output_projection=None,
                                feed_previous=False,
                                update_embedding_for_previous=True,
                                dtype=None,
                                scope=None,
                                initial_state_attention=False):
  """RNN decoder with embedding and attention and a pure-decoding option.
  """

  if output_size is None:
    output_size = cell.output_size
  if output_projection is not None:
    proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
    proj_biases.get_shape().assert_is_compatible_with([num_symbols])

  with variable_scope.variable_scope(
      scope or "embedding_attention_decoder", dtype=dtype) as scope:
    embedding = load_embedding()

    if feed_previous:
      # Inference mode.
      loop_function = local_extract_argmax_and_embed(
          embedding, output_projection,
          update_embedding_for_previous)
    else:
      # Training mode.
      loop_function = None
    emb_inp = [
        embedding_ops.embedding_lookup(embedding, i) for i in decoder_inputs]
    return local_attention_decoder(
        emb_inp,
        initial_state,
        attention_states,
        cell,
        output_size=output_size,
        num_heads=num_heads,
        loop_function=loop_function,
        initial_state_attention=initial_state_attention)


def local_embedding_attention_seq2seq(encoder_inputs,
                                decoder_inputs,
                                cell,
                                num_encoder_symbols,
                                num_decoder_symbols,
                                embedding_size,
                                num_heads=1,
                                output_projection=None,
                                feed_previous=False,
                                dtype=None,
                                scope=None,
                                initial_state_attention=False):
  """Embedding sequence-to-sequence model with attention.
  """
  with variable_scope.variable_scope(
      scope or "embedding_attention_seq2seq", dtype=dtype) as scope:
    dtype = scope.dtype
    # Encoder.
    encoder_cell = LocalEmbeddingWrapper(
        cell, embedding_classes=num_encoder_symbols,
        embedding_size=embedding_size)

    encoder_outputs, encoder_state = rnn.rnn(
        encoder_cell, encoder_inputs, dtype=dtype)


    ### POINTER SENTINEL: encoder_outputs --> need these 

    # First calculate a concatenation of encoder outputs to put attention on.
    top_states = [array_ops.reshape(e, [-1, 1, cell.output_size])
                  for e in encoder_outputs]
    attention_states = array_ops.concat(1, top_states)

    # print("SDFSDFDSFDSFSDFDSFDSF\n\n")
    # print('encoder_outputs: ', encoder_outputs)
    # print('top_states: ', top_states)
    # print('attention_states: ', attention_states)
    # print('')

    # DECODER.
    output_size = None
    assert output_projection
    # if output_projection is None:
    #   cell = rnn_cell.OutputProjectionWrapper(cell, num_decoder_symbols)
    #   output_size = num_decoder_symbols

    # If feed_previous is a boolean, return this.
    if isinstance(feed_previous, bool):
      return local_embedding_attention_decoder(
          decoder_inputs,
          encoder_state,
          attention_states,
          cell,
          num_decoder_symbols,
          embedding_size,
          num_heads=num_heads,
          output_size=output_size,
          output_projection=output_projection,
          feed_previous=feed_previous,
          initial_state_attention=initial_state_attention)

    print("Never go here. Dead code. feed_previous must be a bool in our case.")
    sys.exit()

    # # If feed_previous is a Tensor, we construct 2 graphs and use cond.
    # def decoder(feed_previous_bool):
    #   reuse = None if feed_previous_bool else True
    #   with variable_scope.variable_scope(
    #       variable_scope.get_variable_scope(), reuse=reuse) as scope:
    #     outputs, state = local_embedding_attention_decoder(
    #         decoder_inputs,
    #         encoder_state,
    #         attention_states,
    #         cell,
    #         num_decoder_symbols,
    #         embedding_size,
    #         num_heads=num_heads,
    #         output_size=output_size,
    #         output_projection=output_projection,
    #         feed_previous=feed_previous_bool,
    #         update_embedding_for_previous=False,
    #         initial_state_attention=initial_state_attention)
    #     state_list = [state]
    #     if nest.is_sequence(state):
    #       state_list = nest.flatten(state)
    #     return outputs + state_list
    # outputs_and_state = control_flow_ops.cond(feed_previous,
    #                                           lambda: decoder(True),
    #                                           lambda: decoder(False))
    # outputs_len = len(decoder_inputs)  # Outputs length same as decoder inputs.
    # state_list = outputs_and_state[outputs_len:]
    # state = state_list[0]
    # if nest.is_sequence(encoder_state):
    #   state = nest.pack_sequence_as(structure=encoder_state,
    #                                 flat_sequence=state_list)
    # return outputs_and_state[:outputs_len], state
##############



def local_model_with_buckets(encoder_inputs, decoder_inputs, targets, weights,
                       buckets, seq2seq, softmax_loss_function=None,
                       per_example_loss=False, name=None):
  """Create a sequence-to-sequence model with support for bucketing.

  The seq2seq argument is a function that defines a sequence-to-sequence model,
  e.g., seq2seq = lambda x, y: basic_rnn_seq2seq(x, y, rnn_cell.GRUCell(24))

  Args:
    encoder_inputs: A list of Tensors to feed the encoder; first seq2seq input.
    decoder_inputs: A list of Tensors to feed the decoder; second seq2seq input.
    targets: A list of 1D batch-sized int32 Tensors (desired output sequence).
    weights: List of 1D batch-sized float-Tensors to weight the targets.
    buckets: A list of pairs of (input size, output size) for each bucket.
    seq2seq: A sequence-to-sequence model function; it takes 2 input that
      agree with encoder_inputs and decoder_inputs, and returns a pair
      consisting of outputs and states (as, e.g., basic_rnn_seq2seq).
    softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
    per_example_loss: Boolean. If set, the returned loss will be a batch-sized
      tensor of losses for each sequence in the batch. If unset, it will be
      a scalar with the averaged loss from all examples.
    name: Optional name for this operation, defaults to "model_with_buckets".

  Returns:
    A tuple of the form (outputs, losses), where:
      outputs: The outputs for each bucket. Its j'th element consists of a list
        of 2D Tensors. The shape of output tensors can be either
        [batch_size x output_size] or [batch_size x num_decoder_symbols]
        depending on the seq2seq model used.
      losses: List of scalar Tensors, representing losses for each bucket, or,
        if per_example_loss is set, a list of 1D batch-sized float Tensors.

  Raises:
    ValueError: If length of encoder_inputsut, targets, or weights is smaller
      than the largest (last) bucket.
  """
  if len(encoder_inputs) < buckets[-1][0]:
    raise ValueError("Length of encoder_inputs (%d) must be at least that of la"
                     "st bucket (%d)." % (len(encoder_inputs), buckets[-1][0]))
  if len(targets) < buckets[-1][1]:
    raise ValueError("Length of targets (%d) must be at least that of last"
                     "bucket (%d)." % (len(targets), buckets[-1][1]))
  if len(weights) < buckets[-1][1]:
    raise ValueError("Length of weights (%d) must be at least that of last"
                     "bucket (%d)." % (len(weights), buckets[-1][1]))

  all_inputs = encoder_inputs + decoder_inputs + targets + weights
  losses = []
  outputs = []
  with ops.name_scope(name, "model_with_buckets", all_inputs):
    for j, bucket in enumerate(buckets):
      with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                         reuse=True if j > 0 else None):
        bucket_outputs, _ = seq2seq(encoder_inputs[:bucket[0]],
                                    decoder_inputs[:bucket[1]])
        outputs.append(bucket_outputs)
        if per_example_loss:
          print("oops took this out. Bring back.")
          sys.exit()
        losses.append(seq2seqFile.sequence_loss(
            outputs[-1], targets[:bucket[1]], weights[:bucket[1]],
            softmax_loss_function=softmax_loss_function))

  return outputs, losses


class Seq2SeqModel(object):
  """Sequence-to-sequence model with attention and for multiple buckets.

  This class implements a multi-layer recurrent neural network as encoder,
  and an attention-based decoder. This is the same as the model described in
  this paper: http://arxiv.org/abs/1412.7449 - please look there for details,
  or into the seq2seq library for complete model implementation.
  This class also allows to use GRU cells in addition to LSTM cells, and
  sampled softmax to handle large output vocabulary size. A single-layer
  version of this model, but with bi-directional encoder, was presented in
    http://arxiv.org/abs/1409.0473
  and sampled softmax is described in Section 3 of the following paper.
    http://arxiv.org/abs/1412.2007
  """

  def __init__(self, use_lstm=False, forward_only=False):
    """Create the model.

    Args:
      vocab_size: actual size of vocab.
      buckets: a list of pairs (I, O), where I specifies maximum input length
        that will be processed in that bucket, and O specifies maximum output
        length. Training instances that have inputs longer than I or outputs
        longer than O will be pushed to the next bucket and padded accordingly.
        We assume that the list is sorted, e.g., [(2, 4), (8, 16)].
      size: number of units in each layer of the model.
      num_layers: number of layers in the model.
      max_gradient_norm: gradients will be clipped to maximally this norm.
      batch_size: the size of the batches used during training;
        the model construction is independent of batch_size, so it can be
        changed after initialization if this is convenient, e.g., for decoding.
      learning_rate: learning rate to start with.
      learning_rate_decay_factor: decay learning rate by this much when needed.
      use_lstm: if true, we use LSTM cells instead of GRU cells.
      num_samples: number of samples for sampled softmax.
      forward_only: if set, we do not construct the backward pass in the model.
    """

    # Just read the lengths of the vocab files, easiest way to determine.
    def file_len(fname):
      with open(fname) as f:
          for i, l in enumerate(f):
              pass
      return i + 1

    size = config.layer_size

    self.vocab_size = file_len(config.vocabPath)
    self.buckets = config._buckets
    self.batch_size = config.batch_size
    self.learning_rate = tf.Variable(float(config.learning_rate), trainable=False)
    self.learning_rate_decay_op = self.learning_rate.assign(
        self.learning_rate * config.learning_rate_decay_factor)
    self.global_step = tf.Variable(0, trainable=False)

    # If we use sampled softmax, we need an output projection.
    output_projection = None
    softmax_loss_function = None
    # Sampled softmax only makes sense if we sample less than vocabulary size.
    if config.num_samples > 0 and config.num_samples < self.vocab_size:
      w = tf.get_variable("proj_w", [size, self.vocab_size],initializer=tf.contrib.layers.xavier_initializer())
      w_t = tf.transpose(w)
      b = tf.get_variable("proj_b", [self.vocab_size],initializer=tf.contrib.layers.xavier_initializer())
      output_projection = (w, b)
      if config.useTensorBoard:
        tf.summary.histogram("Output_Projection_W", w)
        tf.summary.histogram("Output_Projection_b", b)

      def sampled_loss(inputs, labels):
        labels = tf.reshape(labels, [-1, 1])
        return tf.nn.sampled_softmax_loss(w_t, b, inputs, labels, config.num_samples,
                self.vocab_size)
      softmax_loss_function = sampled_loss

    # Create the internal multi-layer cell for our RNN.
    single_cell = tf.nn.rnn_cell.GRUCell(size)
    # if use_lstm:
    #   single_cell = tf.nn.rnn_cell.BasicLSTMCell(size)
    single_cell = tf.nn.rnn_cell.DropoutWrapper(single_cell, input_keep_prob=config.dropout_keep)
    cell = single_cell
    if config.num_layers > 1:
      cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * config.num_layers)

    # The seq2seq function: we use embedding for the input and attention.
    def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
      # return local_embedding_attention_seq2seq(
      # return tf.nn.seq2seq.embedding_attention_seq2seq(
      return local_embedding_attention_seq2seq(
          encoder_inputs, decoder_inputs, cell,
          num_encoder_symbols=self.vocab_size,
          num_decoder_symbols=self.vocab_size,
          embedding_size=config.glove_dim,
          output_projection=output_projection,
          feed_previous=do_decode)

    # Feeds for inputs.
    self.encoder_inputs = []
    self.decoder_inputs = []
    self.target_weights = []
    for i in xrange(self.buckets[-1][0]):  # Last bucket is the biggest one.
      self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="encoder{0}".format(i)))
    for i in xrange(self.buckets[-1][1] + 1):
      self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="decoder{0}".format(i)))
      self.target_weights.append(tf.placeholder(tf.float32, shape=[None],
                                                name="weight{0}".format(i)))

    # Our targets are decoder inputs shifted by one.
    targets = [self.decoder_inputs[i + 1]
               for i in xrange(len(self.decoder_inputs) - 1)]







    # Training outputs and losses.
    if forward_only:
      self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(
          self.encoder_inputs, self.decoder_inputs, targets,
          self.target_weights, self.buckets, lambda x, y: seq2seq_f(x, y, True),
          softmax_loss_function=softmax_loss_function)
      # If we use output projection, we need to project outputs for decoding.
      if output_projection is not None:
        for buck in xrange(len(self.buckets)):
          self.outputs[buck] = [
              tf.matmul(output, output_projection[0]) + output_projection[1]
              for output in self.outputs[buck]
          ]
    else:



      # local_model_with_buckets(
      self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(
          self.encoder_inputs, self.decoder_inputs, targets,
          self.target_weights, self.buckets,
          lambda x, y: seq2seq_f(x, y, False),
          softmax_loss_function=softmax_loss_function)

      # print("HERE")
      # print(self.losses)
      # self.losses = tf.Print(self.losses, [self.losses], message="lossesForBuckets: ", first_n=50, summarize=100)
      # tf.summary.histogram('lossesForBuckets', self.losses)


      ######
      # POINTER SENTINEL MIXTURE MODEL.
      # print(self.outputs[0]) ---> shape [batchSize x hiddenStateSize]. Will need these h's. These are the decoder h's.
      # self.encoder_inputs, self.decoder_inputs --> necessary for constructing cost function.
      # model_with_buckets --> ALSO neeed to return encoder_inputs (and if we wish, 'a' for visualization.)





      # print(len(self.outputs[0]))
      # print(len(self.outputs[1]))
      # print(len(self.outputs[2]))
      # print(len(self.outputs[3]))
      # print(len(self.outputs[4]))
      # # 11
      # # 16
      # # 26
      # # 51
      # # 61

    # print("OVER HERE")
    # print(self.losses)

    #self.losses = tf.Print(self.losses, [self.losses], message="lossesForBuckets: ", first_n=50, summarize=100)
    # tf.summary.histogram('lossesForBuckets', self.losses)


    # Gradients and SGD update operation for training the model.
    params = tf.trainable_variables()
    if not forward_only:
      self.gradient_norms = []
      self.updates = []
      # opt = tf.train.GradientDescentOptimizer(self.learning_rate)
      opt = tf.train.AdamOptimizer(self.learning_rate)

      for buck in xrange(len(self.buckets)):
        gradients = tf.gradients(self.losses[buck], params)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                         config.max_gradient_norm)
        self.gradient_norms.append(norm)
        self.updates.append(opt.apply_gradients(
            zip(clipped_gradients, params), global_step=self.global_step))

        # print('gradients: ', gradients)
        # print("modelWithBucketsOutputs_%s" % buck, self.outputs[buck])
        # print("modelWithBucketsLosses_%s" % buck, self.losses[buck])
        # print('gradient_norm_%s' % buck, norm)
        # print('clipped_gradients_%s' % buck, clipped_gradients)
        # sys.exit()

        if config.useTensorBoard and True:
          pass
          # tf.summary.histogram("modelWithBucketsOutputs_%s" % buck, self.outputs[buck])
          # print(self.losses[buck])
          # print(self.losses)
          # tf.summary.scalar("modelWithBucketsLosses_%s" % buck, self.losses[buck])
          # tf.summary.histogram('gradients_%s' % buck, gradients)
          # tf.summary.histogram('gradient_norm_%s' % buck, norm)
          # tf.summary.histogram('clipped_gradients_%s' % buck, clipped_gradients)

    self.saver = tf.train.Saver(tf.global_variables())

  def step(self, session, encoder_inputs, decoder_inputs, target_weights,
           bucket_id, forward_only, summary_op):
    """Run a step of the model feeding the given inputs.

    Args:
      session: tensorflow session to use.
      encoder_inputs: list of numpy int vectors to feed as encoder inputs.
      decoder_inputs: list of numpy int vectors to feed as decoder inputs.
      target_weights: list of numpy float vectors to feed as target weights.
      bucket_id: which bucket of the model to use.
      forward_only: whether to do the backward step or only forward.

    Returns:
      A triple consisting of gradient norm (or None if we did not do backward),
      average perplexity, and the outputs.

    Raises:
      ValueError: if length of encoder_inputs, decoder_inputs, or
        target_weights disagrees with bucket size for the specified bucket_id.
    """
    # Check if the sizes match.
    encoder_size, decoder_size = self.buckets[bucket_id]
    if len(encoder_inputs) != encoder_size:
      raise ValueError("Encoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(encoder_inputs), encoder_size))
    if len(decoder_inputs) != decoder_size:
      raise ValueError("Decoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(decoder_inputs), decoder_size))
    if len(target_weights) != decoder_size:
      raise ValueError("Weights length must be equal to the one in bucket,"
                       " %d != %d." % (len(target_weights), decoder_size))

    # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
    input_feed = {}
    for l in xrange(encoder_size):
      input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
    for l in xrange(decoder_size):
      input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
      input_feed[self.target_weights[l].name] = target_weights[l]

    # Since our targets are decoder inputs shifted by one, we need one more.
    last_target = self.decoder_inputs[decoder_size].name
    input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

    # Output feed: depends on whether we do a backward step or not.
    if not forward_only:
      if config.useTensorBoard:
        # tf.summary.scalar("modelWithBucketsLosses_%s" % bucket_id, self.losses[bucket_id])
        output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                       self.gradient_norms[bucket_id],  # Gradient norm.
                       self.losses[bucket_id], summary_op]  # Loss for this batch.
      else:
        output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                       self.gradient_norms[bucket_id],  # Gradient norm.
                       self.losses[bucket_id]]  # Loss for this batch.
    else:
      output_feed = [self.losses[bucket_id]]  # Loss for this batch.
      for l in xrange(decoder_size):  # Output logits.
        output_feed.append(self.outputs[bucket_id][l])
      # if config.mode == 'test':
      #   global attention_where
      #   output_feed.append(attention_where) ### <---- really want to get this part working.

    outputs = session.run(output_feed, input_feed)
    if not forward_only:
      if config.useTensorBoard:
        return outputs[1], outputs[2], None, outputs[3] # Gradient norm, loss, no outputs, summary_op
      else:
        return outputs[1], outputs[2], None, None # Gradient norm, loss, no outputs
    else:
      return outputs[decoder_size+1:] , outputs[0], outputs[1:decoder_size+1]# attention_where, loss, outputs.

  # get_batch is also called from 'decode', with batch_size one. Places in 
  # correct format.
  def get_batch(self, data, bucket_id, trumpData=None, reducedWeight=None):
    """Get a random batch of data from the specified bucket, prepare for step.

    To feed data in step(..) it must be a list of batch-major vectors, while
    data here contains single length-major cases. So the main logic of this
    function is to re-index data cases to be in the proper format for feeding.

    Args:
      data: a tuple of size len(self.buckets) in which each element contains
        lists of pairs of input and output data that we use to create a batch.
      bucket_id: integer, which bucket to get the batch for.

    Returns:
      The triple (encoder_inputs, decoder_inputs, target_weights) for
      the constructed batch that has the proper format to call step(...) later.
    """
    encoder_size, decoder_size = self.buckets[bucket_id]
    encoder_inputs, decoder_inputs = [], []

    # Get a random batch of encoder and decoder inputs from data,
    # pad them if needed, reverse encoder inputs and add GO to decoder.
    for _ in xrange(self.batch_size):

      indexToInclude = random.choice(range(len(data[bucket_id])))
      # print('indexToInclude ')
      # print(indexToInclude)
      # print('data[bucket_id][indexToInclude] ')
      # print(data[bucket_id][indexToInclude])

      inputs = data[bucket_id][indexToInclude]
      if len(inputs) == 2:
        encoder_input, decoder_input = inputs
        data_weight = 1.0
      else:
        encoder_input, decoder_input, data_weight = inputs

      # print('data_weight')
      # print(data_weight)

      # Encoder inputs are padded and then reversed.
      encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
      # print('encoder input: ', encoder_input)
      # print('reversed(encoder_input): ', reversed(encoder_input))
      # print('list(encoder_pad + reversed(encoder_input)): ', list(encoder_pad + reversed(encoder_input)))
      encoder_inputs.append(encoder_pad + list(reversed(encoder_input)))

      # Decoder inputs get an extra "GO" symbol, and are padded then.
      decoder_pad_size = decoder_size - len(decoder_input) - 1
      decoder_inputs.append([data_utils.GO_ID] + decoder_input +
                            [data_utils.PAD_ID] * decoder_pad_size)

    # Now we create batch-major vectors from the data selected above.
    batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

    # Batch encoder inputs are just re-indexed encoder_inputs.
    for length_idx in xrange(encoder_size):
      batch_encoder_inputs.append(
          np.array([encoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))

    # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
    for length_idx in xrange(decoder_size):
      batch_decoder_inputs.append(
          np.array([decoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))

      # Create target_weights to be 0 for targets that are padding.
      batch_weight = np.ones(self.batch_size, dtype=np.float32) * float(data_weight)
      for batch_idx in xrange(self.batch_size):
        # We set weight to 0 if the corresponding target is a PAD symbol.
        # The corresponding target is decoder_input shifted by 1 forward.
        if length_idx < decoder_size - 1:
          target = decoder_inputs[batch_idx][length_idx + 1]
        if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
          batch_weight[batch_idx] = 0.0
      batch_weights.append(batch_weight)
    return batch_encoder_inputs, batch_decoder_inputs, batch_weights
