"""
Seq2SeqModel in execute.py creates
seq2seq_f which is a function that adds to the graph
  it is called within the initialization of Seq2SeqModel
  that "function" is calls on
embedding_attention_seq2seq:
  this creates an encoder
  passes the encoder_output to a decoder
  and returns results of the decoder.
  The decoder is created by
embedding_attention_decoder:
  this has an option of use either
  a) _extract_beam_search + beam_attention_decoder
  b) _extract_argmax_and_embed + attention_decoder
  The "extract" components return a loop function tells the RNN
    what to do in each time step
  Both those decoders contain similar elements, we go deeper into
beam_attention_decoder:
  this contains it's own attention function
  the creation of beam_path and beam_symbols
beam_symbols holds the ids that are used to lookup the actual word
  in the embedding matrix.
beam_path holds a list of which parent an output came from.  Assuming k = 3,
  and we have already predicted for 5 time steps, I think this means we
  have a list such as: [[0,2,1,2,2],
                        [0,2,1,0,2],
                        [0,2,2,1,1]]
  there are 3 items in the list because at that time step, we only hold
  three candidates. there are 5 items because there have been 5 predictions
  so far.  The numbers only go up to 2, because there are only 3 parents
  possible at each iteration.
"""

    # log_beam_probs: list of [beam_size, 1] Tensors
    #  Ordered log probabilities of the `beam_size` best hypotheses
    #  found in each beam step (highest probability first).
    # beam_symbols: list of [beam_size] Tensors
    #  The ordered `beam_size` words / symbols extracted by the beam
    #  step, which will be appended to their corresponding hypotheses
    #  (corresponding hypotheses found in `beam_path`).
    # beam_path: list of [beam_size] Tensor
    #  The ordered `beam_size` parent indices. Their values range
    #  from [0, `beam_size`), and they denote which previous
    #  hypothesis each word should be appended to.
    log_beam_probs, beam_symbols, beam_path  = [], [], []
    def beam_search(prev, i):
        if output_projection is not None:
            prev = tf.nn.xw_plus_b(
                prev, output_projection[0], output_projection[1])

        # Compute
        #  log P(next_word, hypothesis) =
        #  log P(next_word | hypothesis)*P(hypothesis) =
        #  log P(next_word | hypothesis) + log P(hypothesis)
        # for each hypothesis separately, then join them together
        # on the same tensor dimension to form the example's
        # beam probability distribution:
        # [P(word1, hypothesis1), P(word2, hypothesis1), ...,
        #  P(word1, hypothesis2), P(word2, hypothesis2), ...]

        # If TF had a log_sum_exp operator, then it would be
        # more numerically stable to use:
        #   probs = prev - tf.log_sum_exp(prev, reduction_dims=[1])
        probs = tf.log(tf.nn.softmax(prev))
        # i == 1 corresponds to the input being "<GO>", with
        # uniform prior probability and only the empty hypothesis
        # (each row is a separate example).
        if i > 1:
            probs = tf.reshape(probs + log_beam_probs[-1],
                               [-1, beam_size * num_symbols])

        # Get the top `beam_size` candidates and reshape them such
        # that the number of rows = batch_size * beam_size, which
        # allows us to process each hypothesis independently.
        best_probs, indices = tf.nn.top_k(probs, beam_size)
        indices = tf.stop_gradient(tf.squeeze(tf.reshape(indices, [-1, 1])))
        best_probs = tf.stop_gradient(tf.reshape(best_probs, [-1, 1]))

        symbols = indices % num_symbols # Which word in vocabulary.
        beam_parent = indices // num_symbols # Which hypothesis it came from.

        beam_symbols.append(symbols)
        beam_path.append(beam_parent)
        log_beam_probs.append(best_probs)
        return tf.nn.embedding_lookup(embedding, symbols)

    # Setting up graph.
    inputs = [tf.placeholder(tf.float32, shape=[None, num_symbols])
              for i in range(num_steps)]
    for i in range(num_steps):
        beam_search(inputs[i], i + 1)

    # Running the graph.
    input_vals = [0, 0, 0]
    l = np.log
    eps = -10 # exp(-10) ~= 0

    # These values mimic the distribution of vocabulary words
    # from each hypothesis independently (in log scale since
    # they will be put through exp() in softmax).
    input_vals[0] = np.array([[0, eps, l(2), eps, l(3)]])
    # Step 1 beam hypotheses =
    # (1) Path: [4], prob = log(1 / 2)
    # (2) Path: [2], prob = log(1 / 3)
    # (3) Path: [0], prob = log(1 / 6)

    input_vals[1] = np.array([[l(1.2), 0, 0, l(1.1), 0], # Path [4]
                              [0,   eps, eps, eps, eps], # Path [2]
                              [0,  0,   0,   0,   0]])   # Path [0]
    # Step 2 beam hypotheses =
    # (1) Path: [2, 0], prob = log(1 / 3) + log(1)
    # (2) Path: [4, 0], prob = log(1 / 2) + log(1.2 / 5.3)
    # (3) Path: [4, 3], prob = log(1 / 2) + log(1.1 / 5.3)

    input_vals[2] = np.array([[0,  l(1.1), 0,   0,   0], # Path [2, 0]
                              [eps, 0,   eps, eps, eps], # Path [4, 0]
                              [eps, eps, eps, eps, 0]])  # Path [4, 3]
    # Step 3 beam hypotheses =
    # (1) Path: [4, 0, 1], prob = log(1 / 2) + log(1.2 / 5.3) + log(1)
    # (2) Path: [4, 3, 4], prob = log(1 / 2) + log(1.1 / 5.3) + log(1)
    # (3) Path: [2, 0, 1], prob = log(1 / 3) + log(1) + log(1.1 / 5.1)


inside create_model
  A) creation of function
        loading glove embedding matrix
        complete loading glove embedding matrix

      inside local_decoder
        loading glove embedding matrix
        complete loading glove embedding matrix
  B) creation of function
        loading glove embedding matrix
        complete loading glove embedding matrix
      inside local_decoder
        loading glove embedding matrix
        complete loading glove embedding matrix