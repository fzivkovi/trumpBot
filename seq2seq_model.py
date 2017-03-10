from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import random

from six.moves import zip
from six.moves import xrange  # pylint: disable=redefined-builtin
import sys
import os

from tensorflow.models.rnn.translate import data_utils
import config

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
from localWrapper import LocalEmbeddingWrapper, load_embedding

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
    # Note that gradients will not propagate through the second parameter of
    # embedding_lookup.
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

          a = nn_ops.softmax(s)
          # if config.mode == 'test':
          #   a = tf.Print(a, [a], message="where I'm paying attention: ", first_n=100, summarize=200)

          # Now calculate the attention-weighted vector d.
          d = math_ops.reduce_sum(
              array_ops.reshape(a, [-1, attn_length, 1, 1]) * hidden,
              [1, 2])
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
      # Run the RNN.
      cell_output, state = cell(x, state)
      # Run the attention mechanism.
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

def local_decoder(decoder_inputs, initial_state,
        attention_states, cell, num_symbols, embedding_size,
        num_heads=1, output_size=None, output_projection=None,
        feed_previous=False, update_embedding_for_previous=True,
        dtype=None, scope=None, initial_state_attention=False):
  """ FEED_PREVIOUS = True means INFERENCE mode where we make predictions
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
    embedding = load_embedding()

    if feed_previous:  # Inference mode.
      loop_function = local_extract_argmax_and_embed(
          embedding, output_projection,
          update_embedding_for_previous)
    else:               # Training mode.
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


def local_seq2seq(encoder_inputs, decoder_inputs,
      cell, num_encoder_symbols, num_decoder_symbols, embedding_size,
      num_heads=1, output_projection=None, feed_previous=False,
      dtype=None, scope=None, initial_state_attention=False):

  with variable_scope.variable_scope(
      scope or "embedding_attention_seq2seq", dtype=dtype) as scope:
    dtype = scope.dtype
    # Encoder.
    encoder_cell = LocalEmbeddingWrapper(
        cell, embedding_classes=num_encoder_symbols,
        embedding_size=embedding_size)
    encoder_outputs, encoder_state = rnn.rnn(
        encoder_cell, encoder_inputs, dtype=dtype)

    # First calculate a concatenation of encoder outputs to put attention on.
    top_states = [array_ops.reshape(e, [-1, 1, cell.output_size])
                  for e in encoder_outputs]
    attention_states = array_ops.concat(1, top_states)

    # Decoder.
    output_size = None
    if output_projection is None:
      cell = rnn_cell.OutputProjectionWrapper(cell, num_decoder_symbols)
      output_size = num_decoder_symbols

    if isinstance(feed_previous, bool):
      return local_decoder(
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

    # If feed_previous is a Tensor, we construct 2 graphs and use cond.
    def decoder(feed_previous_bool):
      reuse = None if feed_previous_bool else True
      with variable_scope.variable_scope(
          variable_scope.get_variable_scope(), reuse=reuse) as scope:
        outputs, state = local_decoder(
            decoder_inputs, encoder_state,
            attention_states, cell,
            num_decoder_symbols, embedding_size,
            num_heads=num_heads,
            output_size=output_size,
            output_projection=output_projection,
            feed_previous=feed_previous_bool,
            update_embedding_for_previous=False,
            initial_state_attention=initial_state_attention)
        state_list = [state]
        if nest.is_sequence(state):
          state_list = nest.flatten(state)
        return outputs + state_list

    outputs_and_state = control_flow_ops.cond(feed_previous,
                                              lambda: decoder(True),
                                              lambda: decoder(False))
    outputs_len = len(decoder_inputs)  # Outputs length same as decoder inputs.
    state_list = outputs_and_state[outputs_len:]
    state = state_list[0]
    if nest.is_sequence(encoder_state):
      state = nest.pack_sequence_as(structure=encoder_state,
                                    flat_sequence=state_list)
    return outputs_and_state[:outputs_len], state

class Seq2SeqModel(object):
  def __init__(self, use_lstm=False, forward_only=False):
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
      return local_seq2seq(
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
      # Outputs is a list of tensors, where the tensors have one for each output word
      self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(
          self.encoder_inputs, self.decoder_inputs, targets,
          self.target_weights, self.buckets,
          lambda x, y: seq2seq_f(x, y, False),
          softmax_loss_function=softmax_loss_function)
    #self.losses = tf.Print(self.losses, [self.losses], message="lossesForBuckets: ", first_n=50, summarize=100)
    # tf.summary.histogram('lossesForBuckets', self.losses)

    # Gradients and SGD update operation for training the model.
    params = tf.trainable_variables()
    if not forward_only:
      self.gradient_norms = []
      self.updates = []
      opt = tf.train.GradientDescentOptimizer(self.learning_rate)

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

        if config.useTensorBoard:
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
    """Run a step of the model feeding the given inputs
    Args:
      encoder and decoder_inputs: list of numpy int vectors to feed in
      target_weights: list of numpy float vectors to feed as target weights.
      bucket_id: which bucket of the model to use.
      forward_only: whether to do the backward step or only forward.
    Returns:
      A tuple consisting of gradient norm (or None if we did not do backward),
      average perplexity, and the outputs.
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

    outputs = session.run(output_feed, input_feed)
    if not forward_only:
      if config.useTensorBoard:
        return outputs[1], outputs[2], None, outputs[3] # Gradient norm, loss, no outputs, summary_op
      else:
        return outputs[1], outputs[2], None, None # Gradient norm, loss, no outputs
    else:
      # this is forward_only. Called from decode, and called when data is validation set.
      return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.

  # get_batch is also called from 'decode', with batch_size one. Places in
  # correct format.
  def get_batch(self, data, bucket_id, trumpData=None, reducedWeight=None):
    """Get a random batch of data from the specified bucket, prepare for step.

    To feed data in step(..) it must be a list of batch-major vectors, while
    data here contains single length-major cases. So the main logic of this
    function is to re-index data cases to be in the proper format for feeding.
    ARGS:
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

    # batch_size = self.batch_size
    # bucket_data = data[bucket_id]
    # encoder_data, decoder_data = bucket_data

    # bucket_len = len(encoder_data)
    # remainder = bucket_len % batch_size
    # divisible_bucket_len = bucket_len - remainder

    # indices = np.arange(bucket_len)
    # if shuffle:
    #   np.random.shuffle(indices)
    # for batch_start in np.arange(0, divisible_bucket_len, batch_size)
    #   q_start = minibatch_start
    #   q_end = minibatch_start + minibatch_size
    #   queries = [encoder_data[i] for i in np.arange(q_start, q_end)]

    #   a_start = minibatch_start + data_size
    #   a_end = minibatch_start + minibatch_size + data_size
    #   answers = [decoder_data[i] for i in np.arange(a_start, a_end)]

    #   yield [queries, answers]

    for _ in xrange(self.batch_size):
      indexToInclude = random.choice(range(len(data[bucket_id])))

      inputs = data[bucket_id][indexToInclude]
      if len(inputs) == 2:
        encoder_input, decoder_input = inputs
        data_weight = 1.0
      else:
        encoder_input, decoder_input, data_weight = inputs

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
        # encoder_batch = []
        # for batch_idx in xrange(self.batch_size):
        #   encoder_batch.append(encoder_inputs[batch_idx][length_idx])
        # np.array(thing, dtype=np.int32))

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

def apply_padding(data, bucket_size):
  pass
  # data < bucket_size:
    # add some zeros
  return data