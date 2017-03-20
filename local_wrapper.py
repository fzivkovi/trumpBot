from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import random

from tensorflow.python.ops import seq2seq
from six.moves import zip
from six.moves import xrange  # pylint: disable=redefined-builtin
import sys
import os

import data_utils
import config
import embedding_utils

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest

linear = rnn_cell._linear  # pylint: disable=protected-access

def local_extract_argmax_and_embed(embedding, output_projection=None,
                              update_embedding=True):
  """Get a loop_function that extracts the previous symbol and embeds it.
    embedding: embedding tensor for symbols.
    output_projection: None or a pair (W, B). If provided, each fed previous
      output will first be multiplied by W and added B.
    update_embedding: Boolean; if False, the gradients will not propagate
      through the embeddings.
  Returns: a loop function.
  """
  def loop_function(prev, _):
    if output_projection is not None:
      prev = nn_ops.xw_plus_b(
          prev, output_projection[0], output_projection[1])
    prev_symbol = math_ops.argmax(prev, 1)
  # Note that gradients will not propagate through the second parameter of embedding_lookup.
    emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
    if not update_embedding:
      emb_prev = array_ops.stop_gradient(emb_prev)
    return emb_prev
  return loop_function

def local_attention_decoder(decoder_inputs, initial_state, attention_states,
        cell, output_size, num_heads=1, loop_function=None, dtype=None,
        scope=None, initial_state_attention=False):
  """ Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors of
        shape [batch_size x output_size]. These represent the generated outputs.
      state: The state of each decoder cell the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].
  """
  if not decoder_inputs:
    raise ValueError("Must provide at least 1 input to attention decoder.")
  if attention_states.get_shape()[2].value is None:
    raise ValueError("Shape[2] of attention_states must be known: %s"
                     % attention_states.get_shape())

  with variable_scope.variable_scope("attention_decoder", dtype=dtype) as scope:
    dtype = scope.dtype

    batch_size = array_ops.shape(decoder_inputs[0])[0]  # Needed for reshaping.
    attn_length = attention_states.get_shape()[1].value
    if attn_length is None:
      attn_length = array_ops.shape(attention_states)[1]

    attn_size = attention_states.get_shape()[2].value
    # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
    hidden = array_ops.reshape(
        attention_states, [-1, attn_length, 1, attn_size])
    hidden_features = []
    v = []

    attention_vec_size = attn_size   # represents hidden state size
    if config.attention_type != 'vinyals':
      attention_vec_size = attn_size * config.num_layers # Size of query vectors for attention.

    for a in xrange(num_heads):
      # THIS IS WHERE W COMES FROM.
      k = variable_scope.get_variable("AttnW_%d" % a,
                                      [1, 1, attn_size, attention_vec_size],initializer=tf.contrib.layers.xavier_initializer())
      hidden_features.append(nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
      v.append(
          variable_scope.get_variable("AttnV_%d" % a, [attention_vec_size], initializer=tf.contrib.layers.xavier_initializer()))

    state = initial_state

    def attention(query):
      """attention masks:
        cell_output, new_state = cell(linear(input, prev_attn), prev_state).
      Then, we calculate new attention masks:
        new_attn = softmax(V^T * tanh(W * attention_states + U * new_state))
      and then we calculate the output:
        output = linear(cell_output, new_attn).
      """
      ds = []  # Results of attention reads will be stored here.
      if nest.is_sequence(query):  # If the query is a tuple, flatten it.
        query_list = nest.flatten(query)
        for q in query_list:  # Check that ndims == 2 if specified.
          ndims = q.get_shape().ndims
          if ndims:
            assert ndims == 2
        query = array_ops.concat(1, query_list)
      for a in xrange(num_heads):
        with variable_scope.variable_scope("Attention_%d" % a):

          ###### OUR CODE HERE.
          # Hidden features: shape=(?, 20, 1, 64), batchsize X EncoderBucketSize x numHeads x hiddenStatesDim
          # query.shape=(?, 64) hiddenStatesDim
          if config.attention_type == 'vinyals':
            y = linear(query, attention_vec_size, True)
            y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
            # Attention mask is a softmax of v^T * tanh(...).
            s = math_ops.reduce_sum(
                v[a] * math_ops.tanh(hidden_features[a] + y), [2, 3])
          elif config.attention_type == "luong":
            # attention = softmax( query * W * attention_states )
            # hidden_features = (W * attention_states)
            y = array_ops.reshape(query, [-1, 1, 1, attention_vec_size])
            s = math_ops.reduce_sum(hidden_features[a] * y, [2, 3])
          elif config.attention_type == "bahdanau":
            y = array_ops.reshape(query, [-1, 1, 1, attention_vec_size])
            s = math_ops.reduce_sum(hidden * y, [2, 3])
          a_soft = nn_ops.softmax(s)
          if config.mode == 'test':
            a = tf.Print(a, [a], message="where I'm paying attention: ", first_n=100, summarize=200)

          # Now calculate the attention-weighted vector d.
          d = math_ops.reduce_sum(
              array_ops.reshape(a_soft, [-1, attn_length, 1, 1]) * hidden, [1, 2])
          ds.append(array_ops.reshape(d, [-1, attn_size]))
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
    # Only have decoder_inputs if training.
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
      # Run the RNN then the attention mechanism.
      cell_output, state = cell(x, state)
      if i == 0 and initial_state_attention:
        with variable_scope.variable_scope(variable_scope.get_variable_scope(), reuse=True):
          attns = attention(state)
      else:
        attns = attention(state)

      with variable_scope.variable_scope("AttnOutputProjection"):
        output = linear([cell_output] + attns, output_size, True)
      if loop_function is not None:
        prev = output
      outputs.append(output)

  return outputs, state

def local_decoder(decoder_inputs, initial_state, attention_states, cell, embeds,
        num_symbols, num_heads=1, output_size=None, output_projection=None,
        test_mode=False, update_embedding_for_previous=True,
        dtype=None, scope=None, initial_state_attention=False):
  """ test_mode = True means TEST mode where we make predictions
      The default of False is for TRAINING mode.
    initial_state_attention: If False (default), initial attentions are zero.
      If True, initialize the attentions from the initial state and attention
      states -- useful when we wish to resume decoding from a previously
      stored decoder state and attention states.

  Returns: outputs: A list of the same length as decoder_inputs, each element
    in the list is a 2D Tensors with shape [batch_size x output_size] containing the generated outputs.
      state: The state of each decoder cell at the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].
  """
  if output_size is None:
    output_size = cell.output_size
  if output_projection is not None:
    proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
    proj_biases.get_shape().assert_is_compatible_with([num_symbols])

  with variable_scope.variable_scope(
      scope or "embedding_attention_decoder", dtype=dtype) as scope:
    if test_mode:  # Inference mode.
      loop_function = local_extract_argmax_and_embed(
          embeds, output_projection,
          update_embedding_for_previous)
    else:               # Training mode.
      loop_function = None

    emb_inp = [
        embedding_ops.embedding_lookup(embeds, i) for i in decoder_inputs]

    return local_attention_decoder(
        emb_inp,
        initial_state,
        attention_states,
        cell,
        output_size=output_size,
        num_heads=num_heads,
        loop_function=loop_function,
        initial_state_attention=initial_state_attention)

def local_seq2seq(encoder_inputs, decoder_inputs, cell, embeddings,
      num_encoder_symbols, num_decoder_symbols,
      num_heads=1, output_projection=None, test_mode=False,
      dtype=None, scope=None, initial_state_attention=False):
  with variable_scope.variable_scope(
      scope or "embedding_attention_seq2seq", dtype=dtype) as scope:
    dtype = scope.dtype
    # Encoder.
    encoder_cell = embedding_utils.EmbeddingWrapper(
        cell, embeddings, classes=num_encoder_symbols)

    encoder_outputs, encoder_state = rnn.rnn(
        encoder_cell, encoder_inputs, dtype=dtype)

    # First calculate a concatenation of encoder outputs to put attention on.
    top_states = [array_ops.reshape(e, [-1, 1, cell.output_size])
                  for e in encoder_outputs]
    attention_states = array_ops.concat(1, top_states)

    # Decoder.
    output_size = None
    assert output_projection
    # if output_projection is None:
    #   cell = rnn_cell.OutputProjectionWrapper(cell, num_decoder_symbols)
    #   output_size = num_decoder_symbols
    if isinstance(test_mode, bool):
      return local_decoder(
          decoder_inputs,
          encoder_state,
          attention_states,
          cell, embeddings,
          num_decoder_symbols,
          num_heads=num_heads,
          output_size=output_size,
          output_projection=output_projection,
          test_mode=test_mode,
          initial_state_attention=initial_state_attention)

def local_model_with_buckets(encoder_inputs, decoder_inputs, targets, weights,
                  buckets, seq2seq_f, softmax_loss_function=None, name=None):
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
    embeddings = embedding_utils.load_vocab()
    for j, bucket in enumerate(buckets):
      print("Preparing bucket", str(j), "...")
      with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                         reuse=True if j > 0 else None):
        bucket_outputs, _ = seq2seq_f(encoder_inputs[:bucket[0]],
                          decoder_inputs[:bucket[1]], embeddings)
        outputs.append(bucket_outputs)
        losses.append(seq2seq.sequence_loss(
            outputs[-1], targets[:bucket[1]], weights[:bucket[1]],
            softmax_loss_function=softmax_loss_function))

  return outputs, losses