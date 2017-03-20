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
| medium | ?      | ????? | ?????  |  ????? --> RUN THIS!
| large  | ?      | ????? | ?????? |  ?????
The exact results may vary depending on the random initialization.


Results using small, num_steps = 100, batchsize=4.
Epoch: 6 Train Perplexity: 54.825
Epoch: 6 Valid Perplexity: 120.714
Test Perplexity: 114.427


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

# import reader
import reader_ptr_sent as reader
import sys
import operator

import os

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", 'saveGeneric',
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_bool("use_wiki_text", False,
                  "Expect wiki-text data in train path, not PTB.")
flags.DEFINE_bool("test", False,
                  "Evaluate just test perplexity alone on best model in save path.")
flags.DEFINE_bool("vis", True, "Return visualization. Extra calculation.")
FLAGS = flags.FLAGS

from tensorflow.python.ops import rnn

from prettytable import PrettyTable


def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32


class PTBInput(object):
  """The input data."""

  def __init__(self, config, data, name=None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
    self.input_data, self.targets = reader.ptb_producer(
        data, batch_size, num_steps, config.L, name=name)


class PTBModel(object):
  """The PTB model."""

  def __init__(self, is_training, config, input_, vis):
    self._input = input_

    batch_size = input_.batch_size
    num_steps = input_.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size


    def lstm_cell():
      return tf.contrib.rnn.BasicLSTMCell(
          size, forget_bias=1.0, state_is_tuple=True)
    attn_cell = lstm_cell
    if is_training and config.keep_prob < 1:
      def attn_cell(i):
        # Different dropout configurations for the layers. As decribed in Pointer Sentinel paper.
        if i == config.num_layers-1:
          return tf.contrib.rnn.DropoutWrapper(
              lstm_cell(), output_keep_prob=config.keep_prob, input_keep_prob=config.keep_prob)
        else:
           return tf.contrib.rnn.DropoutWrapper(
              lstm_cell(), input_keep_prob=config.keep_prob)         

      cell = tf.contrib.rnn.MultiRNNCell(
          [attn_cell(i) for i in range(config.num_layers)], state_is_tuple=True)

    else:
      cell = tf.contrib.rnn.MultiRNNCell(
          [attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)

    self._initial_state = cell.zero_state(batch_size, data_type())

    with tf.device("/cpu:0"):
      embedding = tf.get_variable(
          "embedding", [vocab_size, size], dtype=data_type())
      inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

    if is_training and config.keep_prob_words < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob_words) 


    outputs_all, state = tf.nn.dynamic_rnn(cell, inputs, 
                               initial_state=self._initial_state) # sequence_length can also be set.

    # The old way to do this:
    # outputs_all = []
    # state = self._initial_state
    # with tf.variable_scope("RNN"):
    #   for time_step in range(num_steps + config.L):
    #     if time_step > 0: tf.get_variable_scope().reuse_variables()
    #     (cell_output, state) = cell(inputs[:, time_step, :], state)
    #     outputs_all.append(cell_output)

    # This is the order it was in for the first method, which I'd written it for.
    outputs_all = tf.transpose(outputs_all,perm=[1,0,2])

    outputs_prediction = tf.gather(outputs_all, tf.range(config.L, config.L+num_steps)) 
    outputs_L = tf.gather(outputs_all, tf.range(0, config.L))

    outputs_all = tf.reshape(tf.concat(outputs_all, 1), [-1, size])
    outputs_prediction = tf.reshape(tf.concat(outputs_prediction, 1), [-1, size])
    # outputs_all --> [step0Batch0Hidden, step0Batch1Hidden, ... step1Batch0Hidden, ...stepNbatchNHidden]
    softmax_w = tf.get_variable(
        "softmax_w", [size, vocab_size], dtype=data_type())
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
    logits = tf.matmul(outputs_prediction, softmax_w) + softmax_b

    #########################################################
    ## Can't use sparse_softmax_cross_entropy_with_logits, 
    ## because cross-entropy needs to be calculated after summation.
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
    ##      don't care for. In the paper, Merity uses the term output and hidden state interchangably.


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
      size_num_steps = tf.one_hot(newIndicies, vocab_size,dtype=data_type())
      # execute multiplication to insert the values.
      originalShape = tf.shape(size_num_steps)
      size_num_steps = tf.reshape(size_num_steps, [-1, vocab_size])
      values = tf.reshape(values, [-1])
      r = tf.transpose(tf.multiply(tf.transpose(size_num_steps),values))
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
    # q calculation is intentionally this way, multiply by every single output,
    # because num_steps * batch_size predictions, so that is how 
    # many q's we require.
    q = tf.tanh(tf.matmul(outputs_prediction, W_for_q) + b_for_q, name='q')

    s = tf.get_variable("s", [size,1], dtype=data_type())
    # s undergoes reduction STEPS*BATCHSIZE X size --> STEPS*BATCHSIZE to result in g's. 
    g = tf.matmul(q,s) # STEPS*BATCHSIZE

    tf.Print(g, [g], message='g ', summarize=100)

    #########################################################
    ## This section is for the L portion.
    ## Calculate pointer outputs, z. [zi = inner(q, hi)] concat with [q*s]. 
    #########################################################
    outputs_L = tf.reshape(tf.concat(outputs_L, 1), [config.L, batch_size, size])
    outputs_L = tf.transpose(outputs_L, perm=[1,0,2])
    q_for_L = tf.reshape(q, [num_steps, batch_size, size])
    q_for_L = tf.transpose(q_for_L, perm=[1,2,0])
    # print(q_for_L) # batchsizex200xnum_steps
    # print(outputs_L) # batchSizexLx200
    z_for_L = tf.matmul(outputs_L, q_for_L) # batchsize x L x num_steps

    #########################################################
    ## This section is for the prediction portion.
    ## Calculate pointer outputs, z. [zi = inner(q, hi)] concat with [q*s]. 
    #########################################################

    z_i = tf.reduce_sum(tf.multiply(outputs_prediction, q), 1, keep_dims=True)

    # Cast to new size --> STEPS*BATCHSIZE x num_seps
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
    finalMasks = tf.reshape(masks, [num_steps*batch_size, num_steps])
    masks = concatenateColumnOntoMatrix(finalMasks, tf.ones_like(g, dtype=data_type()), num_steps, batch_size)

    # Must only grab the inputs we are making predictions on.
    inp = input_.input_data
    input_num_steps_range = tf.transpose(tf.gather(tf.transpose(inp), tf.range(config.L,num_steps+config.L)))

    # Indexes to place numbers when casting to vocab size.
    inputMapping = tf.map_fn(lambda x: getLowerDiag(x), input_num_steps_range)
    inputMapping = tf.transpose(inputMapping, perm=[1, 0, 2]) 
    inputMapping = tf.reshape(inputMapping, [num_steps*batch_size, num_steps])

    #########################################################
    ## Concatenate sentinels for L with those from num_steps
    #########################################################

    # print(z) # num_steps*batch_size x (num_steps + 1)
    # print(masks)#  num_steps*batch_size x (num_steps + 1)
    # print(inputMapping)#  num_steps*batch_size x num_steps 
    # print(z_for_L) # batchsize x L x num_steps
  
    z_for_L = tf.transpose(z_for_L, perm=[2,0,1])
    z_for_L = tf.reshape(z_for_L, [num_steps*batch_size, config.L])
    z = tf.concat([z_for_L, z],1)

    input_L_range = tf.transpose(tf.gather(tf.transpose(inp), tf.range(config.L)))
    input_L_range = tf.tile(input_L_range, [num_steps, 1])
    # print(input_L_range) # num_steps*batch_size x L
    inputMapping = tf.concat([input_L_range, inputMapping],1)

    masks_for_L = tf.ones_like(input_L_range, dtype=data_type())
    masks = tf.concat([masks_for_L, masks],1)

    #########################################################
    ## Calculate masked softmax for p_ptr, transform to sparse matrix 
    #########################################################

    # Do the softmax on z. Awesome trick.
    z_softmaxed = tf.nn.softmax(tf.log(masks) + z)
    # Take g out.
    p_ptr_dense, g = splitOffG(z_softmaxed)
    masks, __ = splitOffG(masks)

    # Return p_ptr of size [step_size*batch_size x vocab_size]
    p_ptr = returnSparse(p_ptr_dense, masks, inputMapping, vocab_size)

    #########################################################
    ## p_final = g * p_vocab + (1 - g) * p_ptr, then apply X-entropy
    #########################################################

    # shouldn't need to transpose twice, revisit if time.
    pointer_contrib = tf.transpose(tf.multiply(tf.transpose(p_ptr), (1-g)))
    vocab_contrib = tf.transpose(tf.multiply(tf.transpose(p_vocab), g))
    p_final = pointer_contrib + vocab_contrib

    # print('input data, ',input_.input_data)
    # print('targets, ',input_.targets)
    targets = tf.reshape(input_.targets, [-1])

    # Calculate cross entropy.
    target_mask = tf.one_hot(targets, vocab_size,dtype=data_type())
    loss = tf.reduce_sum(target_mask * -tf.log(p_final), 1)
    self._cost = cost = tf.reduce_sum(loss) / batch_size
    self._final_state = state

    if vis:
      self._g_s = g 
      self._in = input_.input_data 
      self._targets = tf.reshape(input_.targets, [-1])
      self._p_ptrs =  p_ptr_dense

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
    #optimizer = tf.train.AdamOptimizer(self.lr)
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
  def g_s(self):
    return self._g_s

  @property
  def inputs_individual(self):
    return self._in

  @property
  def targets(self):
    return self._targets

  @property
  def p_ptrs(self):
    return self._p_ptrs


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
  # learning_rate = 0.008
  learning_rate = 1.0
  max_grad_norm = 1
  num_layers = 1
  num_steps = 2
  hidden_size = 4
  max_epoch = 1
  max_max_epoch = 2
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 40
  vocab_size = 10000
  L = 10
  keep_prob_words = 1

class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  # learning_rate = 0.001
  learning_rate = 1.0
  max_grad_norm = 1
  num_layers = 1 # Change this to two later.
  num_steps = 1 # Change to 20 later.
  L = 80
  hidden_size = 350
  max_epoch = 4
  max_max_epoch = 25
  keep_prob = 0.5
  lr_decay = 0.5
  batch_size = 40
  vocab_size = 10000
  keep_prob_words = 0.5

class MediumConfig(object):
  """Medium config."""
  init_scale = 0.05
  #learning_rate = 0.003
  learning_rate = 1.0
  max_grad_norm = 1
  num_layers = 2
  num_steps = 25
  L = 85
  hidden_size = 500
  max_epoch = 6
  max_max_epoch = 80
  keep_prob = 0.5
  keep_prob_words = 0.5
  lr_decay = 0.5
  batch_size = 5
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
  L = 100


def run_epoch(session, model, eval_op=None, verbose=False, ids_to_words=None):
  """Runs the model on the given data."""
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = session.run(model.initial_state)


  if FLAGS.vis:
    fetches = {
        "cost": model.cost,
        "final_state": model.final_state,
        "Gs": model.g_s,
        "in": model.inputs_individual,
        "targets": model.targets,
        "p_ptr": model.p_ptrs,
    }
  else:
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

    if FLAGS.vis and ids_to_words:
      Gs = vals["Gs"]
      inputs = vals["in"]
      targets = vals["targets"]
      p_ptr = vals["p_ptr"]

      # print('gs ', Gs)
      # print('inputs ',inputs)
      # print('targets ',targets)
      # print('p_ptr ', p_ptr)

      correspondingIndex, minG = min(enumerate(Gs), key=operator.itemgetter(1))
      with open('gValues.txt','a') as f:
        # So that we monitor see whether g is being used.
        f.write(str(minG)[:3]+'\n')

      if minG < 0.4:

        # Chop off those that were masked, and format inputs.
        inputs = [[i]* model.input.num_steps for i in inputs] # FIGURE OUT WHICH TO REMOVE.
        inputs = [i for numsteps in inputs for i in numsteps]

        # print(correspondingIndex)
        # print(inputs)

        minGInputs = [ids_to_words[i] for i in inputs[correspondingIndex]]
        minGTargets= ids_to_words[targets[correspondingIndex]]
        minG_p_ptr = p_ptr[correspondingIndex]

        if minGTargets in minGInputs:
          correctIndexToPoint = minGInputs.index(minGTargets)
          predictedIndex = list(minG_p_ptr).index(max(minG_p_ptr))
          max_value = max(minG_p_ptr)
          predicted_value = minG_p_ptr[predictedIndex]
          diff = max_value - predicted_value
          nextMaxPredictedPercent = max(n for n in minG_p_ptr if n!=max_value)
          correct = False
          if correctIndexToPoint == predictedIndex:
            result = "Correctly done! Points at '%s' with percentage %s" % (minGInputs[predictedIndex], minG_p_ptr[predictedIndex])
            correct = True
          else:
            result = "Points at '%s' but should be '%s', diff %s" % (minGInputs[predictedIndex], minGInputs[correctIndexToPoint] , diff)

          if (correct == True and nextMaxPredictedPercent > 0.1) or (correct==False and diff > 0.1):
            with open('visualizations.txt', 'a') as f:
              f.write("A GOOD DATA POINT HERE: \n\n")

          t = PrettyTable(['Parameter', 'Values']) # could make prettier by separating each value.
          t.add_row(['g', minG])
          t.add_row(['inputs', minGInputs])
          t.add_row(['p_ptr', minG_p_ptr])
          t.add_row(['targetWord', minGTargets])
          t.add_row(['result', result])
          with open('visualizations.txt', 'a') as f:
            f.write(str(t) + '\n\n')

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
  train_data, valid_data, test_data, _, word_to_ids = raw_data

  if not FLAGS.vis:
    ids_to_words = None
  else:
    ids_to_words = {v: k for k, v in word_to_ids.iteritems()}

  if not os.path.exists(FLAGS.save_path):
      os.makedirs(FLAGS.save_path)

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
        m = PTBModel(is_training=True, config=config, vis=FLAGS.vis, input_=train_input)
      tf.summary.scalar("Training Loss", m.cost)
      tf.summary.scalar("Learning Rate", m.lr)

    with tf.name_scope("Valid"):
      valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mvalid = PTBModel(is_training=False, config=config, vis=FLAGS.vis,input_=valid_input)
      tf.summary.scalar("Validation Loss", mvalid.cost)

    with tf.name_scope("Test"):
      test_input = PTBInput(config=eval_config, data=test_data, name="TestInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mtest = PTBModel(is_training=False, config=eval_config, vis=FLAGS.vis, 
                         input_=test_input)


    # Now begins a new session and saves it along the way.
    # Automatically resumes session if in save path.
    bestRunningValidationPerplexity = 1000
    sv = tf.train.Supervisor(logdir=FLAGS.save_path)
    cproto = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    with sv.managed_session(config=cproto) as session:
      for i in range(config.max_max_epoch):
        lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
        m.assign_lr(session, config.learning_rate * lr_decay)

        print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
        train_perplexity = run_epoch(session, m, eval_op=m.train_op,
                                     verbose=True, ids_to_words=ids_to_words)

        print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
        valid_perplexity = run_epoch(session, mvalid, ids_to_words=ids_to_words)
        print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

        # Save only if better than any before.
        if FLAGS.save_path and bestRunningValidationPerplexity >= valid_perplexity:
          print("Saving model to %s." % FLAGS.save_path)
          #global_step = "epoch_%s_valid_perp_%s" % (i,valid_perplexity)
          sv.saver.save(session, os.path.join(FLAGS.save_path, 'epoch.txt'), global_step=i)

        if FLAGS.test:
          test_perplexity = run_epoch(session, mtest,ids_to_words=ids_to_words)
          print("Test Perplexity: %.3f" % test_perplexity)

if __name__ == "__main__":
  tf.app.run()
