
Welcome to Pointer Sentinel Portion of the project.
This portion requires Tensorflow version 1.0.

####################################
## Reproducing my results        ###
####################################

The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:
$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz

To execute (Zaremba, et. al.) without modifications, run:
$ python ptb_word_lm_original.py --data_path=simple-examples/data/

To execute Pointer Sentinel with Decoder Length of 1, run this:
$ python pointer_sent_test1.py --data_path=simple-examples/data/ --save_path=test1 --model=official_test_1

To execute Pointer Sentinel with Decoder Length > 1, run this:
$ python pointer_sent_test2.py --data_path=simple-examples/data/ --save_path=test2 --model=official_test_2

Seeing Errors? 
--> If modifying hyperparameters, make sure not to accidentally load old configurations. 
--> Did you install the correct version of tensorflow.

##################
## Description ###
##################

The starter code for this is was in models/tutorials/rnn/ptb in the TensorFlow models repo.
Tutorial is here: https://www.tensorflow.org/tutorials/recurrent
Original Mode:
  (Zaremba, et. al.) Recurrent Neural Network Regularization
  http://arxiv.org/abs/1409.2329
Our Adaptation:
  Pointer Sentinel Mixtrue Model
  https://arxiv.org/abs/1609.07843

##################
## Results     ###
##################

============================================================
Model Of Choice  | config | epochs | train | valid  | test |
============================================================
Zaremba, et. al. | small  | 13     | 37.99 | 121.39 | 115.91
Pointer Sentinel | small  | 15...     | 


Pointer Sentinel
Add three to epochs. Tests still runnign....
Epoch: 8 Train Perplexity: 62.350
Epoch: 8 Valid Perplexity: 96.972
Saving model to hopefullyWorksOfficial1.
Test Perplexity: 93.177




