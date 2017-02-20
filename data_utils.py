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
print('loading en for spacy')
nlp = spacy.load('en')
print('complete loading en for spacy')


from six.moves import urllib

from tensorflow.python.platform import gfile

# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
hyperlink = b"https:<link>"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK, hyperlink]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3
hyperlink = 4

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")

import sys
reload(sys)  
sys.setdefaultencoding('utf8')

def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    if isinstance(space_separated_fragment, str):
        print(space_separated_fragment)
        word = str.encode(space_separated_fragment, errors='ignore')
    else:
        word = space_separated_fragment  
    words.extend(re.split(_WORD_SPLIT, word))

  return [w for w in words if w]

def spacy_tokenizer(paragraph, encoder=True):
  words = []
  intermediate_words = []

  def processWord(word):
    word = str(word).lower().strip()
    if 'http' in word:
      return hyperlink
    return word

  doc1 = nlp(unicode(paragraph, errors='ignore'))
  for sent in doc1.sents:
    sentence = [processWord(word) for word in sent]
    sentence = [str(word)  for word in sentence if word]
    words.extend(sentence)
    intermediate_words.append(sentence)

  # Make sure that none are larger than our largest bucket.
  while len(words) >= 50:
    if encoder:
      del intermediate_words[0]
    else:
      del intermediate_words[-1]
    words = [w for s in intermediate_words for w in s]

  return words


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      tokenizer=None, normalize_digits=True):
    
  if not gfile.Exists(vocabulary_path):
    print("Creating vocabulary %s from %s" % (vocabulary_path, data_path))
    vocab = {}

    # Support reading from multiple files.
    if type(data_path) == str:
      data_path = [data_path]
    for one_data_path in data_path:
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
    if len(vocab_list) > max_vocabulary_size:
      vocab_list = vocab_list[:max_vocabulary_size]
    with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
      for i,w in enumerate(vocab_list):
        vocab_file.write(w + b"\n")


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



def prepare_custom_data(working_directory, train_enc, train_trump_enc, train_trump_dec, train_dec, test_enc, test_dec, enc_vocabulary_size, dec_vocabulary_size, tokenizer=None):

    tokenizer = spacy_tokenizer

    # Create vocabularies of the appropriate sizes.
    enc_vocab_path = os.path.join(working_directory, "vocab%d.enc" % enc_vocabulary_size)
    dec_vocab_path = os.path.join(working_directory, "vocab%d.dec" % dec_vocabulary_size)
    create_vocabulary(enc_vocab_path, [train_trump_enc, train_enc], enc_vocabulary_size, tokenizer)
    create_vocabulary(dec_vocab_path, [train_trump_dec, train_dec], dec_vocabulary_size, tokenizer)

    # Create token ids for the training data.
    enc_train_ids_path = train_enc + (".ids%d" % enc_vocabulary_size)
    dec_train_ids_path = train_dec + (".ids%d" % dec_vocabulary_size)
    enc_train_trump_ids_path = train_trump_enc + (".ids%d" % enc_vocabulary_size)
    dec_train_trump_ids_path = train_trump_dec + (".ids%d" % enc_vocabulary_size)
    data_to_token_ids(train_enc, enc_train_ids_path, enc_vocab_path, tokenizer)
    data_to_token_ids(train_dec, dec_train_ids_path, dec_vocab_path, tokenizer)
    data_to_token_ids(train_trump_enc, enc_train_trump_ids_path, enc_vocab_path, tokenizer)
    data_to_token_ids(train_trump_dec, dec_train_trump_ids_path, dec_vocab_path, tokenizer)

    # Create token ids for the development data.
    enc_dev_ids_path = test_enc + (".ids%d" % enc_vocabulary_size)
    dec_dev_ids_path = test_dec + (".ids%d" % dec_vocabulary_size)
    data_to_token_ids(test_enc, enc_dev_ids_path, enc_vocab_path, tokenizer)
    data_to_token_ids(test_dec, dec_dev_ids_path, dec_vocab_path, tokenizer)

    return (enc_train_ids_path, enc_train_trump_ids_path, dec_train_trump_ids_path, dec_train_ids_path, enc_dev_ids_path, dec_dev_ids_path, enc_vocab_path, dec_vocab_path)







