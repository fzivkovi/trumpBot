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

import data_utils
import config
from local_wrapper import *

class Seq2SeqModel(object):
  def __init__(self):
    size = config.layer_size
    self.vocab_size = data_utils.get_vocab_length(config.vocabPath)
    self.buckets = config._buckets
    self.mode = config.mode
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
    cell = tf.nn.rnn_cell.GRUCell(size)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=config.dropout_keep)
    if config.num_layers > 1:
      cell = tf.nn.rnn_cell.MultiRNNCell([cell] * config.num_layers)

    # The seq2seq function: we use embedding for the input and attention.
    def seq2seq_f(encoder_inputs, decoder_inputs, embeddings, mode):
      make_predictions = True if mode == "test" else False
      return local_seq2seq(
          encoder_inputs, decoder_inputs, cell, embeddings,
          num_encoder_symbols=self.vocab_size,
          num_decoder_symbols=self.vocab_size,
          output_projection=output_projection,
          test_mode=make_predictions, )

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

    self.outputs, self.losses = local_model_with_buckets(
        self.encoder_inputs, self.decoder_inputs, targets,
        self.target_weights, self.buckets,
        lambda x, y, embeds: seq2seq_f(x, y, embeds, self.mode),
        softmax_loss_function=softmax_loss_function)
    # If we use output projection, we need to project outputs for decoding.
    if self.mode == "test" and output_projection is not None:
      for buck in xrange(len(self.buckets)):
        self.outputs[buck] = [
            tf.matmul(output, output_projection[0]) + output_projection[1]
            for output in self.outputs[buck]
        ]
    # Outputs is a list of tensors, where the tensors have one for each output word
    ######
    # POINTER SENTINEL MIXTURE MODEL.
    # print(self.outputs[0]) ---> shape [batchSize x hiddenStateSize]. Will need these h's. These are the decoder h's.
    # self.encoder_inputs, self.decoder_inputs --> necessary for constructing cost function.
    # model_with_buckets --> ALSO neeed to return encoder_inputs (and if we wish, 'a' for visualization.)

    #self.losses = tf.Print(self.losses, [self.losses], message="lossesForBuckets: ", first_n=50, summarize=100)
    # tf.summary.histogram('lossesForBuckets', self.losses)

    # Gradients and SGD update operation for training the model.
    params = tf.trainable_variables()
    if self.mode == "train":
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

        if config.useTensorBoard:
          pass
          # tf.summary.histogram("modelWithBucketsOutputs_%s" % buck, self.outputs[buck])
          # print(self.losses[buck])
          # print(self.losses)
          # tf.summary.scalar("modelWithBucketsLosses_%s" % buck, self.losses[buck])
          # tf.summary.histogram('gradients_%s' % buck, gradients)
          # tf.summary.histogram('gradient_norm_%s' % buck, norm)
          # tf.summary.histogram('clipped_gradients_%s' % buck, clipped_gradients)

    self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

  def step(self, session, encoder_inputs, decoder_inputs, target_weights,
           bucket_id, test_mode, summary_op):
    """Run a step of the model feeding the given inputs
    Args:
      encoder and decoder_inputs: list of numpy int vectors to feed in
      target_weights: list of numpy float vectors to feed as target weights.
      bucket_id: which bucket of the model to use.
      test_mode: True if we are testing, False if we are training
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
    if test_mode:
      output_feed = [self.losses[bucket_id]]  # Loss for this batch.
      for l in xrange(decoder_size):  # Output logits.
        output_feed.append(self.outputs[bucket_id][l])
      # if config.mode == 'test':
      #   global attention_where
      #   output_feed.append(attention_where) ### <---- really want to get this part working.
    else:
      if config.useTensorBoard:
        # tf.summary.scalar("modelWithBucketsLosses_%s" % bucket_id, self.losses[bucket_id])
        output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                       self.gradient_norms[bucket_id],  # Gradient norm.
                       self.losses[bucket_id], summary_op]  # Loss for this batch.
      else:
        output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                       self.gradient_norms[bucket_id],  # Gradient norm.
                       self.losses[bucket_id]]  # Loss for this batch.

    outputs = session.run(output_feed, input_feed)
    if test_mode:
      return outputs[decoder_size+1:] , outputs[0], outputs[1:decoder_size+1]# attention_where, loss, outputs.
    else:
      if config.useTensorBoard:
        return outputs[1], outputs[2], None, outputs[3] # Gradient norm, loss, no outputs, summary_op
      else:
        return outputs[1], outputs[2], None, None # Gradient norm, loss, no outputs

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

    for i in xrange(self.batch_size):
      indexToInclude = random.choice(range(len(data[bucket_id])))
      inputs = data[bucket_id][indexToInclude]

      if len(inputs) == 2:
        encoder_input, decoder_input = inputs
        data_weight = 1.0
      else:
        print("GOOD IN HERE YAY. Just had to test this.. remove now.")
        sys.exit()
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
