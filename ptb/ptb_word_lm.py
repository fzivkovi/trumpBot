"""

Filip Zivkovic, Derek Chen
CS224N, 2017, One Piece of Final Project.
Isolated test of Pointer Sentinel Mixture Model on Penn Treebank data.

#################################
## Grabbing the data and run ####
#################################

Dependencies: Tensorflow Version 1.0

The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:
$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz
WikiText dataset:
$ wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip


To run:

$ python ptb_word_lm.py --data_path=simple-examples/data/

##################
## Description ###
##################

Executed an isolated test. Only change: adding pointer sentinel.
Intent is to analyze how this one change effects results. 

The starter code for this is was in models/tutorials/rnn/ptb in the TensorFlow models repo.
Tutorial is here: https://www.tensorflow.org/tutorials/recurrent
Original Mode:
  (Zaremba, et. al.) Recurrent Neural Network Regularization
  http://arxiv.org/abs/1409.2329
Adaptation:
  Pointer Sentinel Mixtrue Model
  https://arxiv.org/abs/1609.07843

#################################
#### Original Models Results  ###
#################################

Example / benchmark for building a PTB LSTM model.

Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.

#################################
#### Pointer Sentinel Results  ##
#################################


Epoch: 16 Train Perplexity: 40.052
Epoch: 16 Valid Perplexity: 118.827
Test Perplexity: 115.282

===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 16     | 40.05 | 118.82 | 115.28 --> should have run more epochs.
| medium | ?      | ????? | ?????  |  ?????
| large  | ?      | ????? | ?????? |  ?????
The exact results may vary depending on the random initialization.


#################################
#### Hyperparameters, unchanged #
#################################

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf

import reader
import sys

import os

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_bool("use_wiki_text", False,
                  "Expect wiki-text data in train path, not PTB.")
flags.DEFINE_bool("test", False,
                  "Evaluate just test perplexity alone on best model in save path.")
FLAGS = flags.FLAGS


def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32


class PTBInput(object):
  """The input data."""

  def __init__(self, config, data, name=None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
    self.input_data, self.targets = reader.ptb_producer(
        data, batch_size, num_steps, name=name)


class PTBModel(object):
  """The PTB model."""

  def __init__(self, is_training, config, input_):
    self._input = input_

    batch_size = input_.batch_size
    num_steps = input_.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size

    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    def lstm_cell():
      return tf.contrib.rnn.BasicLSTMCell(
          size, forget_bias=0.0, state_is_tuple=True)
    attn_cell = lstm_cell
    if is_training and config.keep_prob < 1:
      def attn_cell():
        return tf.contrib.rnn.DropoutWrapper(
            lstm_cell(), output_keep_prob=config.keep_prob)
    cell = tf.contrib.rnn.MultiRNNCell(
        [attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)

    self._initial_state = cell.zero_state(batch_size, data_type())

    with tf.device("/cpu:0"):
      embedding = tf.get_variable(
          "embedding", [vocab_size, size], dtype=data_type())
      inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob) ### QUESTION: WHY DROPOUT ON INPUT DATA?

    # Simplified version of models/tutorials/rnn/rnn.py's rnn().
    # This builds an unrolled LSTM for tutorial purposes only.
    # In general, use the rnn() or state_saving_rnn() from rnn.py.
    #
    # The alternative version of the code below is:
    #
    # inputs = tf.unstack(inputs, num=num_steps, axis=1)
    # outputs, state = tf.nn.rnn(cell, inputs,
    #                            initial_state=self._initial_state)
    outputs = []
    state = self._initial_state
    with tf.variable_scope("RNN"):
      for time_step in range(num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)

    output = tf.reshape(tf.concat(outputs, 1), [-1, size])
    # output --> [step0Batch0Hidden, step0Batch1Hidden, ... step1Batch0Hidden, ...stepNbatchNHidden]
    softmax_w = tf.get_variable(
        "softmax_w", [size, vocab_size], dtype=data_type())
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
    logits = tf.matmul(output, softmax_w) + softmax_b


    #########################################################
    ## Can't use sparse_softmax_cross_entropy_with_logits, 
    ## because cross-entropy needs to be calcualted after summation.
    #########################################################
    p_vocab = tf.nn.softmax(logits)


    #################################
    #### Common QA for code #########
    #################################

    #### QUESTION: How many predictions is this code making?
    #### ANSWER: target is simply input shifted by one. Thus we are actually making BATCH_SIZE*STEPS predictions each batch.
    # print('outputs ', outputs) # list STEPS , tensor BATCHSIZE x HIDDEN_STATE_SIZE 
    # print('output ', output) # STEPS*BATCHSIZE x HIDDEN_STATE_SIZE
    # print('logits ', logits) ## STEPS*BATCHSIZE x VOCABSIZE
    # Even TARGET is of size STEPS*BATCHSIZE.
    #### NEXT QUESTION: does the previous answer imply that the first prediction only has access to 
    #                   one hidden word? Whereas the later ones have access to many?
    #### Answer: yes. But that means it transfers to my chatbot easier. 
    #            We test this scenario. So I like it like this.
    ## Questions: how come we're using "outputs" in our calculation, weren't we supposed to use hidden state?
    ## Answer: They are the same. The "state" variable returns hidden state as well as cell state, which we 
    ##      don't care for. In the paper, Steven Merity uses the term output and hidden state interchangably.


    #################################
    #### BEGIN POINTER SENTINEL #####
    #################################

    # Can only use previous hidden states in the prediction.
    # Tiles data and then returns lower-diagonal matrix.
    # Because of this, the length of "L" varies.
    def getLowerDiag(inputs):
      inputs_matrix = tf.reshape(tf.tile(inputs, [tf.shape(inputs)[0]]), [-1,tf.shape(inputs)[0]])
      result = tf.matrix_band_part(inputs_matrix, -1, 0)
      return result

    def concatenateColumnOntoMatrix(z, g, num_steps, batch_size):
      # Creativity. Done by multiplying values of g with one-hot column in sparse matrix, and adding
      # the two matricies together.
      z_with_pad = tf.pad(z, [[0,0],[0,1]], mode='CONSTANT', name=None)
      sparseMatrix = tf.one_hot(tf.tile([num_steps],[num_steps*batch_size]), num_steps+1, dtype=tf.float32)
      result = sparseMatrix * g + z_with_pad
      return result

    def splitOffG(myMatrix):
      # Removes the last column from the matrix, returns them seperately.
      g = tf.gather(tf.transpose(myMatrix), tf.shape(myMatrix)[1]-1)
      z = tf.gather(tf.transpose(myMatrix), tf.range(tf.shape(myMatrix)[1]-1))
      return tf.transpose(z),g

    def returnSparse(values, mask, indices, vocab_size):
      ########
      # EXAMPLE:
      # Input parameters:
      # values = tf.constant([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.2, 0.5, 0.3]])
      # mask = tf.constant([[1, 0, 0], [1, 1, 0], [1, 1, 1]])
      # indices = tf.constant([[5, 0, 0], [2, 3, 0], [3, 7, 8]])
      # vocab_size = 10
      # Returns:
      # [array([[ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
      #          1.        ,  0.        ,  0.        ,  0.        ,  0.        ],
      #        [ 0.        ,  0.        ,  0.5       ,  0.5       ,  0.        ,
      #          0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
      #        [ 0.        ,  0.        ,  0.        ,  0.2       ,  0.        ,
      #          0.        ,  0.        ,  0.5       ,  0.30000001,  0.        ]], dtype=float32)]
      #########
      # shift 1's-->0's, 0's--> -1's. 
      # This adds 0 to the indicies that matter, and adds -1 to the empty ones.
      newIndicies = (tf.to_int32(mask) - 1) + indices

      # LxV
      sizeL = tf.one_hot(newIndicies, vocab_size,dtype=data_type())
      # execute multiplication to insert the values.
      originalShape = tf.shape(sizeL)
      sizeL = tf.reshape(sizeL, [-1, vocab_size])
      values = tf.reshape(values, [-1])
      r = tf.transpose(tf.multiply(tf.transpose(sizeL),values))
      r = tf.reshape(r,originalShape)
      # reduce extra L dimensions. 
      r = tf.reduce_sum(r, 1)
      return r

    #########################################################
    ## Calculate q, g. Trainable parameters: W_for_q, b_for_q, s
    #########################################################

    # Pointer Sentinel: calculate q's. q = tanh(W*(last_output) + b)
    W_for_q = tf.get_variable("W_for_q", [size, size], dtype=data_type())
    b_for_q = tf.get_variable("b_for_q", [size], dtype=data_type())
    q = tf.tanh(tf.matmul(output, W_for_q) + b_for_q, name='q')

    sentinel = tf.get_variable("s", [size,1], dtype=data_type())
    # sentinel undergoes reduction STEPS*BATCHSIZE X size --> STEPS*BATCHSIZE to result in g's. 
    g = tf.matmul(q,sentinel) # STEPS*BATCHSIZE

    #########################################################
    ## Calculate pointer outputs, z. [zi = inner(q, hi)] concat with [q*s]. 
    #########################################################

    z_i = tf.reduce_sum(tf.multiply(output, q), 1, keep_dims=True)

    # Cast to new size --> STEPS*BATCHSIZE x L
    z_shape_of_input = tf.reshape(z_i, [batch_size, num_steps])
    z_dense = tf.map_fn(lambda x: getLowerDiag(x), z_shape_of_input)
    z_dense = tf.transpose(z_dense, perm=[1, 0, 2]) 
    z_dense = tf.reshape(z_dense, [num_steps*batch_size, num_steps])
    # append the g before softmax.
    z = concatenateColumnOntoMatrix(z_dense, g, num_steps, batch_size)

    # Grab masks for z_dense
    masks = tf.ones([batch_size, num_steps])
    masks = tf.map_fn(lambda x: getLowerDiag(x), masks)
    masks = tf.transpose(masks, perm=[1, 0, 2]) 
    masks = tf.reshape(masks, [num_steps*batch_size, num_steps])
    masks = concatenateColumnOntoMatrix(masks, tf.ones_like(g, dtype=data_type()), num_steps, batch_size)

    #########################################################
    ## Calculate masked softmax for p_ptr, transform to sparse matrix 
    #########################################################

    # Do the softmax on z. Awesome trick.
    z_softmaxed = tf.nn.softmax(tf.log(masks) + z)
    # Take g out.
    p_ptr_dense, g = splitOffG(z_softmaxed)
    masks, __ = splitOffG(masks)

    # Indexes to place numbers when casting to vocab size.
    inputMapping = tf.map_fn(lambda x: getLowerDiag(x), input_.input_data)
    inputMapping = tf.transpose(inputMapping, perm=[1, 0, 2]) 
    inputMapping = tf.reshape(inputMapping, [num_steps*batch_size, num_steps])

    # Return p_ptr of size [step_size*batch_size x vocab_size]
    p_ptr = returnSparse(p_ptr_dense, masks, inputMapping, vocab_size)

    #########################################################
    ## p = g * p_vocab + (1 - g) * p_train, then apply X-entropy
    #########################################################

    pointer_contrib = tf.transpose(tf.multiply(tf.transpose(p_ptr), (1-g)))
    vocab_contrib = tf.transpose(tf.multiply(tf.transpose(p_vocab), g))

    p_final = pointer_contrib + vocab_contrib
    # print('input data, ',input_.input_data)
    # print('targets, ',input_.targets)
    targets = tf.reshape(input_.targets, [-1])

    # Calculate loss by creating a one-hot mask (target), multiply, then reduce_sum along that axis.
    target_mask = tf.one_hot(targets, vocab_size,dtype=data_type())
    after_mask = tf.reduce_sum(target_mask * p_final, 1)
    loss = -tf.log(after_mask)
    self._cost = cost = tf.reduce_sum(loss) / batch_size
    self._final_state = state

    #################################
    ####   END POINTER SENTINEL #####
    #################################

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self._lr)
    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.contrib.framework.get_or_create_global_step())

    self._new_lr = tf.placeholder(
        data_type(), shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  @property
  def input(self):
    return self._input

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op



# Just use for testing purposes.
class TinyConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 1
  num_layers = 1
  num_steps = 2
  hidden_size = 2
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000


class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 16
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000


class MediumConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 35
  hidden_size = 650
  max_epoch = 6
  max_max_epoch = 39
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 20
  vocab_size = 10000


class LargeConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  num_steps = 35
  hidden_size = 1500
  max_epoch = 14
  max_max_epoch = 55
  keep_prob = 0.35
  lr_decay = 1 / 1.15
  batch_size = 20
  vocab_size = 10000


def run_epoch(session, model, eval_op=None, verbose=False):
  """Runs the model on the given data."""
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = session.run(model.initial_state)

  fetches = {
      "cost": model.cost,
      "final_state": model.final_state,
  }
  if eval_op is not None:
    fetches["eval_op"] = eval_op

  for step in range(model.input.epoch_size):
    feed_dict = {}
    for i, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[i].c
      feed_dict[h] = state[i].h

    vals = session.run(fetches, feed_dict)
    cost = vals["cost"]
    state = vals["final_state"]

    costs += cost
    iters += model.input.num_steps

    if verbose and step % (model.input.epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
             iters * model.input.batch_size / (time.time() - start_time)))

  return np.exp(costs / iters)


def get_config():
  if FLAGS.model == "small":
    return SmallConfig()
  elif FLAGS.model == "medium":
    return MediumConfig()
  elif FLAGS.model == "large":
    return LargeConfig()
  elif FLAGS.model == "tiny":
    return TinyConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)


def main(_):
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")

  raw_data = reader.ptb_raw_data(FLAGS.data_path)
  train_data, valid_data, test_data, _ = raw_data

  config = get_config()
  eval_config = get_config()
  # eval_config.batch_size = 1
  # eval_config.num_steps = 1

  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)

    with tf.name_scope("Train"):
      train_input = PTBInput(config=config, data=train_data, name="TrainInput")
      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m = PTBModel(is_training=True, config=config, input_=train_input)
      tf.summary.scalar("Training Loss", m.cost)
      tf.summary.scalar("Learning Rate", m.lr)

    with tf.name_scope("Valid"):
      valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mvalid = PTBModel(is_training=False, config=config, input_=valid_input)
      tf.summary.scalar("Validation Loss", mvalid.cost)

    with tf.name_scope("Test"):
      test_input = PTBInput(config=eval_config, data=test_data, name="TestInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mtest = PTBModel(is_training=False, config=eval_config,
                         input_=test_input)

    # TODO: This doesn't work. sad.
    # if FLAGS.test:
    #   # Load model, run epoch on test data, print test perplexity, quit
    #   saver = tf.train.Saver()
    #   sv = tf.train.Supervisor(logdir=FLAGS.save_path)
    #   with sv.managed_session() as session:
    #     # Restore variables from disk.
    #     print("Loading best model.")
    #     sv.saver.restore(session, FLAGS.save_path)
    #     test_perplexity = run_epoch(session, mtest)
    #     print("Test Perplexity: %.3f" % test_perplexity)
    #   sys.exit()

    # small TODO: make so that you can resume training a session as well.
    # Now begins a new session and saves it along the way.
    bestRunningValidationPerplexity = 1000
    sv = tf.train.Supervisor(logdir=FLAGS.save_path)
    with sv.managed_session() as session:
      for i in range(config.max_max_epoch):
        lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
        m.assign_lr(session, config.learning_rate * lr_decay)

        print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
        train_perplexity = run_epoch(session, m, eval_op=m.train_op,
                                     verbose=True)

        print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
        valid_perplexity = run_epoch(session, mvalid)
        print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

        # Save only if better than any before.
        if FLAGS.save_path and bestRunningValidationPerplexity >= valid_perplexity:
          print("Saving model to %s." % FLAGS.save_path)
          #global_step = "epoch_%s_valid_perp_%s" % (i,valid_perplexity)
          sv.saver.save(session, FLAGS.save_path, global_step=i)#, max_to_keep=None)

        if FLAGS.test:
          test_perplexity = run_epoch(session, mtest)
          print("Test Perplexity: %.3f" % test_perplexity)

if __name__ == "__main__":
  tf.app.run()
