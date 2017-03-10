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

"""Utilities for downloading data from WMT, tokenizing, vocabularies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import spacy
import math
import numpy as np
import config
from tqdm import *


from six.moves import urllib

from tensorflow.python.platform import gfile

# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
hyperlink = b"https:<link>"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK, hyperlink]

from tensorflow.models.rnn.translate import data_utils
PAD_ID = data_utils.PAD_ID # 0
GO_ID = data_utils.GO_ID # 1
EOS_ID = data_utils.EOS_ID # 2
UNK_ID = data_utils.UNK_ID # 3
hyperlink = 4

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")

import sys
reload(sys)
sys.setdefaultencoding('utf8')


nlp = None
def load_en():
  global nlp
  if not nlp:
    print('loading en for spacy')
    nlp = spacy.load('en')
    print('complete loading en for spacy')
  return nlp

def spacy_tokenizer(paragraph):
    # Uses spacy to parse multi-sentence paragraphs into a list of token words.
    words = []
    intermediate_words = []

    def processWord(word):
        word = str(word).lower().strip()
        if 'http' in word:
            return [hyperlink]
        # Don't clean these!
        if word in [',', '.', '?', '!', ':', '"', '\'', '/', '[', ']', '+', '-', '--', '&', '@']:
            return [word]
        m = re.findall('([\w\.\,\:\%\$\']+)', word)
        m = [w.strip('.') for w in m]
        m = [w for w in m if w]
        return m

    nlp = load_en()
    doc1 = nlp(unicode(paragraph, errors='ignore'))
    for sent in doc1.sents:
        # 2-d list.
        sentence = [processWord(word) for word in sent]
        # flatten.
        sentence = [word for words in sentence for word in words]
        sentence = [str(word) for word in sentence if word]
        words.extend(sentence)
        intermediate_words.append(sentence)

    # Make sure that none are larger than our largest bucket.
    while len(words) >= config.max_sentence_word_count:
        del intermediate_words[-1]
        words = [w for s in intermediate_words for w in s]

    return words


def create_vocabulary(vocabulary_path, data_path, normalize_digits=True):
    
  tokenizer = spacy_tokenizer
  vocab_list = None
  if not gfile.Exists(vocabulary_path):
    print("Creating vocabulary %s from %s" % (vocabulary_path, data_path))
    vocab = {}

    # Support reading from multiple files.
    if type(data_path) == str:
      data_path = [data_path]
    for one_data_path in data_path:
      print(one_data_path)
      if not one_data_path:
        continue
      with gfile.GFile(one_data_path, mode="rb") as f:
        counter = 0
        for line in f:
          counter += 1
          if counter % 100000 == 0:
            print("  processing line %d" % counter)
          tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
          for w in tokens:
            word = re.sub(_DIGIT_RE, b"0", w) if normalize_digits else w
            if word in vocab:
              vocab[word] += 1
            else:
              vocab[word] = 1

    # Drops the words that had least occurences.
    vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
    print('>> Full Vocabulary Size :',len(vocab_list))
    # We don't care for max-vocabulary. 
    # if len(vocab_list) > max_vocabulary_size:
    #   vocab_list = vocab_list[:max_vocabulary_size]
    with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
      for i,w in enumerate(vocab_list):
        vocab_file.write(w + b"\n")

  return vocab_list

def initialize_vocabulary(vocabulary_path):

  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="rb") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])

    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary, tokenizer=None, normalize_digits=True):

  tokenizer = spacy_tokenizer

  if tokenizer:
    words = tokenizer(sentence)
  else:
    words = basic_tokenizer(sentence)
  if not normalize_digits:
    return [vocabulary.get(w, UNK_ID) for w in words]
  # Normalize digits by 0 before looking words up in the vocabulary.
  return [vocabulary.get(re.sub(_DIGIT_RE, b"0", w), UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=True):

  tokenizer = spacy_tokenizer

  if not data_path:
    return

  if not gfile.Exists(target_path):
    print("Tokenizing data in %s" % data_path)
    vocab, _ = initialize_vocabulary(vocabulary_path)
    with gfile.GFile(data_path, mode="rb") as data_file:
      with gfile.GFile(target_path, mode="w") as tokens_file:
        counter = 0
        for line in data_file:
          counter += 1
          if counter % 100000 == 0:
            print("  tokenizing line %d" % counter)
          token_ids = sentence_to_token_ids(line, vocab, tokenizer,
                                            normalize_digits)
          tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")

def process_glove(vocab_list, gloveSize=4e5):
  """
  :param vocab_list: [vocab]
  :return:
  """

  # how to load afterwards:
  # gloveNpz = np.load(config.glove_word_embeddings_path + '.npz','rb')
  # all_data['embedding_matrix'] = gloveNpz['glove']

  if not gfile.Exists(config.glove_word_embeddings_path + ".npz") or vocab_list:

    if not vocab_list:
      vocab_list = []
      with open(config.vocabPath, 'r') as file:
        for line in file:
          line = line.strip()
          vocab_list.append(line)

    glove_path = os.path.join(config.glove_dir, "glove.6B.{}d.txt".format(config.glove_dim))
    # DEREK: These are initialized to zero vectors. 
    # If not found in glove, they remain zero vectors, and are not overwritten.
    # TODO: investigate other options. Try training them as well.
    glove = np.zeros((len(vocab_list), config.glove_dim)) ### FIX.

    vocabNotFound = []
    not_found = 0
    with open(glove_path, 'r') as fh:
      for line in tqdm(fh, total=gloveSize):
          array = line.lstrip().rstrip().split(" ")
          word = array[0]
          vector = list(map(float, array[1:]))
          if word in vocab_list:
              idx = vocab_list.index(word)
              glove[idx, :] = vector
          elif word.capitalize() in vocab_list:
              idx = vocab_list.index(word.capitalize())
              glove[idx, :] = vector
          elif word.lower() in vocab_list:
              idx = vocab_list.index(word.lower())
              glove[idx, :] = vector
          elif word.upper() in vocab_list:
              idx = vocab_list.index(word.upper())
              glove[idx, :] = vector
          else:
              not_found += 1

    for i, vocabWord in enumerate(vocab_list):
      if sum(glove[i,:]) == 0:
        vocabNotFound.append(vocabWord)
        glove[i,:] = np.random.uniform(-math.sqrt(3),math.sqrt(3),config.glove_dim)

    print(vocabNotFound)
    found = gloveSize - not_found
    print("{}/{} of word vocab have corresponding vectors in {}".format(found, len(vocab_list), glove_path))
    np.savez_compressed(config.glove_word_embeddings_path, glove=glove)
    print("saved trimmed glove matrix at: {}".format(config.glove_word_embeddings_path))



def prepare_custom_data():

    tokenizer = spacy_tokenizer

    # Create vocabularies of the appropriate sizes.
    vocab_list = create_vocabulary(config.vocabPath, [config.train_movie_enc, config.train_enc, config.train_movie_dec, config.train_dec])
    process_glove(vocab_list)

    # Create token ids for the training data.
    data_to_token_ids(config.train_enc, config.id_file_train_enc, config.vocabPath, tokenizer)
    data_to_token_ids(config.train_dec, config.id_file_train_dec, config.vocabPath, tokenizer)
    if config.train_movie_enc and config.train_movie_dec:
      data_to_token_ids(config.train_movie_enc, config.id_file_train_movie_enc, config.vocabPath, tokenizer)
      data_to_token_ids(config.train_movie_dec, config.id_file_train_movie_dec, config.vocabPath, tokenizer)

    # Create token ids for the development data.
    data_to_token_ids(config.dev_enc, config.dev_enc_path, config.vocabPath, tokenizer)
    data_to_token_ids(config.dev_dec, config.dev_dec_path, config.vocabPath, tokenizer)

    return







