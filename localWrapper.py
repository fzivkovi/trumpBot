# We overloaded EmbeddingWrapper, filled it with the glove embeddings
# rather than the default random initialization.
import tensorflow as tf
import config
import data_utils
import math
import sys
import numpy as np

from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import embedding_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops

def load_embedding():
  print('loading glove embedding matrix')
  gloveNpz = np.load(config.glove_word_embeddings_path + '.npz','rb')
  embedding_matrix = tf.constant(gloveNpz['glove'], tf.float32)
  print('complete loading glove embedding matrix')
  return embedding_matrix

class LocalEmbeddingWrapper(tf.nn.rnn_cell.RNNCell):
  """Operator adding input embedding to the given cell.

  Note: in many cases it may be more efficient to not use this wrapper,
  but instead concatenate the whole sequence of your inputs in time,
  do the embedding on this batch-concatenated sequence, then split it and
  feed into your RNN.
  """

  def __init__(self, cell, embedding_classes, embedding_size, initializer=None):
    """Create a cell with an added input embedding.

    Args:
      cell: an RNNCell, an embedding will be put before its inputs.
      embedding_classes: integer, how many symbols will be embedded.
      embedding_size: integer, the size of the vectors we embed into.
      initializer: an initializer to use when creating the embedding;
        if None, the initializer from variable scope or a default one is used.

    Raises:
      TypeError: if cell is not an RNNCell.
      ValueError: if embedding_classes is not positive.
    """
    if not isinstance(cell, tf.nn.rnn_cell.RNNCell):
      raise TypeError("The parameter cell is not RNNCell.")
    if embedding_classes <= 0 or embedding_size <= 0:
      raise ValueError("Both embedding_classes and embedding_size must be > 0: "
                       "%d, %d." % (embedding_classes, embedding_size))
    self._cell = cell
    self._embedding_classes = embedding_classes
    self._embedding_size = embedding_size
    self._initializer = initializer
    self.embedding_matrix = load_embedding()

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size

  def __call__(self, inputs, state, scope=None):
    """Run the cell on embedded inputs."""

    with vs.variable_scope(scope or type(self).__name__):  # "EmbeddingWrapper"
      with ops.device("/cpu:0"):

        if type(state) is tuple:
          data_type = state[0].dtype
        else:
          data_type = state.dtype

        embedded = embedding_ops.embedding_lookup(
            self.embedding_matrix, array_ops.reshape(inputs, [-1]))
    return self._cell(embedded, state)





