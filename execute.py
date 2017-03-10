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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import data_utils
import seq2seq_model

import config


def read_data():
  """Read data from source and target files and put into buckets.
    Could definitely have been done neater but oh well. MVP.
  """

  source_path = config.id_file_train_enc
  target_path = config.id_file_train_dec
  movie_source_path = config.id_file_train_movie_enc
  movie_target_path = config.id_file_train_movie_dec
  dev_source_path = config.id_file_dev_enc
  dev_target_path = config.id_file_dev_dec

  data_set = [[] for _ in config._buckets]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      source, target = source_file.readline(), target_file.readline()
      counter = 0
      while source and target:
        counter += 1
        if counter % 100000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()
        source_ids = [int(x) for x in source.split()]
        target_ids = [int(x) for x in target.split()]
        target_ids.append(data_utils.EOS_ID)
        for bucket_id, (source_size, target_size) in enumerate(config._buckets):
          if len(source_ids) < source_size and len(target_ids) < target_size:
            if movie_source_path and movie_target_path and config.reduced_weight:
              data_set[bucket_id].append([source_ids, target_ids, 1.0])
            else:
              data_set[bucket_id].append([source_ids, target_ids])
            break
        source, target = source_file.readline(), target_file.readline()
  if movie_source_path and movie_target_path and config.reduced_weight:
    with tf.gfile.GFile(movie_source_path, mode="r") as source_file:
      with tf.gfile.GFile(movie_target_path, mode="r") as target_file:
        source, target = source_file.readline(), target_file.readline()
        counter = 0
        while source and target:
          counter += 1
          if counter % 100000 == 0:
            print("  reading data line %d" % counter)
            sys.stdout.flush()
          source_ids = [int(x) for x in source.split()]
          target_ids = [int(x) for x in target.split()]
          target_ids.append(data_utils.EOS_ID)
          for bucket_id, (source_size, target_size) in enumerate(config._buckets):
            if len(source_ids) < source_size and len(target_ids) < target_size:
              data_set[bucket_id].append([source_ids, target_ids, config.reduced_weight])
              break
          source, target = source_file.readline(), target_file.readline()
  merged_train_set = data_set

  data_set = [[] for _ in config._buckets]
  with tf.gfile.GFile(dev_source_path, mode="r") as source_file:
    with tf.gfile.GFile(dev_target_path, mode="r") as target_file:
      source, target = source_file.readline(), target_file.readline()
      counter = 0
      while source and target:
        counter += 1
        if counter % 100000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()
        source_ids = [int(x) for x in source.split()]
        target_ids = [int(x) for x in target.split()]
        target_ids.append(data_utils.EOS_ID)
        for bucket_id, (source_size, target_size) in enumerate(config._buckets):
          if len(source_ids) < source_size and len(target_ids) < target_size:
            if movie_source_path and movie_target_path and config.reduced_weight:
              data_set[bucket_id].append([source_ids, target_ids, 1.0])
            else:
              data_set[bucket_id].append([source_ids, target_ids])
            break
        source, target = source_file.readline(), target_file.readline()
    dev_set = data_set

  return dev_set, merged_train_set


def create_model(session, forward_only):

  """Create model and initialize or load parameters"""
  model = seq2seq_model.Seq2SeqModel(forward_only=forward_only)

  # Should fix....
  # if 'pretrained_model' in gConfig:
  #     model.saver.restore(session,gConfig['pretrained_model'])
  #     return model

  ckpt = tf.train.get_checkpoint_state(config.working_directory)
  if ckpt and ckpt.model_checkpoint_path:
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.global_variables_initializer())

  return model


def train():
  # prepare dataset
  print("Preparing data in %s" % config.working_directory)
  data_utils.prepare_custom_data()

  # setup config to use BFC allocator
  config_tf = tf.ConfigProto()
  config_tf.gpu_options.allocator_type = 'BFC'

  with tf.Session(config=config_tf) as sess:
    # Create model.
    print("Creating %d layers of %d units." % (config.num_layers, config.layer_size))
    model = create_model(sess, False)

    if config.useTensorBoard:
      summary_op = tf.summary.merge_all()
      writer = tf.summary.FileWriter(config.logs_path, graph=tf.get_default_graph())
    else:
      summary_op = None

    # Read data into buckets and compute their sizes.
    print ("Reading development and training data (limit: %d)."
           % config.max_train_data_size)

    dev_set, merged_train_set = read_data()

    train_bucket_sizes = [len(merged_train_set[b]) for b in xrange(len(config._buckets))]
    train_total_size = float(sum(train_bucket_sizes))

    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]

    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    while True:
      # Choose a bucket according to data distribution. We pick a random number
      # in [0, 1] and use the corresponding interval in train_buckets_scale.
      random_number_01 = np.random.random_sample()
      bucket_id = min([i for i in xrange(len(train_buckets_scale))
                       if train_buckets_scale[i] > random_number_01])

      # Get a batch and make a step.
      start_time = time.time()
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          merged_train_set, bucket_id)
      _, step_loss, _, tb_summary = model.step(sess, encoder_inputs, decoder_inputs,
                                   target_weights, bucket_id, False, summary_op)
      step_time += (time.time() - start_time) / config.steps_per_checkpoint
      loss += step_loss / config.steps_per_checkpoint
      current_step += 1

      # if current_step == 1:
      #   summary_op = tf.summary.merge_all()

      # Once in a while, we save checkpoint, print statistics, and run evals.
      if current_step % config.steps_per_checkpoint == 0:
        
        if config.useTensorBoard:
          writer.add_summary(tb_summary, current_step)

        # Print statistics for the previous epoch.
        perplexity = math.exp(loss) if loss < 300 else float('inf')
        print ("global step %d learning rate %.4f step-time %.2f perplexity "
               "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                         step_time, perplexity))
        # Decrease learning rate not necessary, using AdamOptimizer.
        # # Decrease learning rate if no improvement was seen over last 3 times.
        # if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
        #   sess.run(model.learning_rate_decay_op)
        previous_losses.append(loss)
        # Save checkpoint and zero timer and loss.
        checkpoint_path = os.path.join(config.working_directory, "seq2seq.ckpt")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        step_time, loss = 0.0, 0.0
        # Run evals on development set and print their perplexity.
        for bucket_id in xrange(len(config._buckets)):
          if len(dev_set[bucket_id]) == 0:
            print("  eval: empty bucket %d" % (bucket_id))
            continue
          encoder_inputs, decoder_inputs, target_weights = model.get_batch(
              dev_set, bucket_id)
          _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True, summary_op)
          eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
          print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
        sys.stdout.flush()


def decode():
  data_utils.load_en()
  with tf.Session() as sess:
    # Create model and load parameters.
    model = create_model(sess, True)
    model.batch_size = 1  # We decode one sentence at a time.

    # Load vocabularies.
    vocab_word_to_id, vocab_list = data_utils.initialize_vocabulary(config.vocabPath)

    # Decode from standard input.
    sys.stdout.write("> ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    while sentence:
      # Get token-ids for the input sentence.
      token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), vocab_word_to_id)
      # Which bucket does it belong to?
      # print('Length token ids:')
      # print(len(token_ids))
      print(' '.join([vocab_list[token_id] for token_id in token_ids]))
      potentialBuckets = [b for b in xrange(len(config._buckets))
                       if config._buckets[b][0] > len(token_ids)]
      if not potentialBuckets:
        print("Too long of input. Continuing.")
        print("> ", end="")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        continue
      bucket_id = min(potentialBuckets)
      # Get a 1-element batch to feed the sentence to the model.
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          {bucket_id: [(token_ids, [])]}, bucket_id)
      # Get output logits for the sentence.
      attention_where, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True, None)
      
      # print(attention_where)
      # # sys.exit()

      # This is a greedy decoder - outputs are just argmaxes of output_logits.
      outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
      print('Untrimmed greedy outputs: %s' % ([tf.compat.as_str(vocab_list[output]) for output in outputs]))

      # If there is an EOS symbol in outputs, cut them at that point.
      if data_utils.EOS_ID in outputs:
        outputs = outputs[:outputs.index(data_utils.EOS_ID)]
      # Print out French sentence corresponding to outputs.
      print("\nTrumps response: ")
      print(" ".join([tf.compat.as_str(vocab_list[output]) for output in outputs]))
      print("> ", end="")
      sys.stdout.flush()
      sentence = sys.stdin.readline()


def self_test():
  """Test the translation model."""
  with tf.Session() as sess:
    print("Self-test for neural translation model.")
    # Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
    model = seq2seq_model.Seq2SeqModel(10, 10, [(3, 3), (6, 6)], 32, 2,
                                       5.0, 32, 0.3, 0.99, num_samples=8)
    sess.run(tf.initialize_all_variables())

    # Fake data set for both the (3, 3) and (6, 6) bucket.
    data_set = ([([1, 1], [2, 2]), ([3, 3], [4]), ([5], [6])],
                [([1, 1, 1, 1, 1], [2, 2, 2, 2, 2]), ([3, 3, 3], [5, 6])])
    for _ in xrange(5):  # Train the fake model for 5 steps.
      bucket_id = random.choice([0, 1])
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          data_set, bucket_id)
      model.step(sess, encoder_inputs, decoder_inputs, target_weights,
                 bucket_id, False, None) # Need to fix "None" in this case.


def init_session(sess):

    # Create model and load parameters.
    model = create_model(sess, True)
    model.batch_size = 1  # We decode one sentence at a time.

    # Load vocabularies.
    enc_vocab, rev_dec_vocab = data_utils.initialize_vocabulary(config.vocabPath)

    return sess, model, enc_vocab, rev_dec_vocab

def decode_line(sess, model, enc_vocab, rev_dec_vocab, sentence):
    # Get token-ids for the input sentence.
    token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), enc_vocab)

    # Which bucket does it belong to?
    bucket_id = min([b for b in xrange(len(config._buckets)) if config._buckets[b][0] > len(token_ids)])

    # Get a 1-element batch to feed the sentence to the model.
    encoder_inputs, decoder_inputs, target_weights = model.get_batch({bucket_id: [(token_ids, [])]}, bucket_id)

    # Get output logits for the sentence.
    _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True, None)

    # This is a greedy decoder - outputs are just argmaxes of output_logits.
    outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]

    # If there is an EOS symbol in outputs, cut them at that point.
    if data_utils.EOS_ID in outputs:
        outputs = outputs[:outputs.index(data_utils.EOS_ID)]

    return " ".join([tf.compat.as_str(rev_dec_vocab[output]) for output in outputs])

if __name__ == '__main__':

    print('\n>> Mode : %s\n' %(config.mode))

    if config.mode == 'train':
        # start training
        train()
    elif config.mode == 'test':
        # interactive decode
        decode()
    else:
        # wrong way to execute "serve"
        #   Use : >> python ui/app.py
        #           uses seq2seq_serve.ini as conf file
        print('Serve Usage : >> python ui/app.py')
        print('# uses seq2seq_serve.ini as conf file')
