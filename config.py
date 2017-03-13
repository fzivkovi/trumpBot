# my config file.
import os
import argparse
import sys

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attention_type'  , '-a', help="Types of attention. Pick from 'vinyals', 'luong', 'bahdanau'", required=True)
    parser.add_argument('--mode'  , '-m', help="What do you want to do? 'train', 'test'.", required=True)
    parser.add_argument('--full'  , '-f', help='Run a full test, not a debug test.', action='store_true')
    args = parser.parse_args(sys.argv[1:])
    return args

args  = parseArguments()
debug = False if args.full else True

attention_type = args.attention_type
assert attention_type in ['vinyals', 'luong', 'bahdanau']
mode = args.mode
assert mode in ['train', 'test', 'serve']

useTensorBoard = False
useMovieData = False
# command is: tensorboard --logdir=run1:/tmp/tensorflow/trump1_luong --port 6006
# useful: tensorboard --inspect --logdir=/tmp/tensorflow/trump3_vinyals
logs_path = '/tmp/tensorflow/trump3_%s' % attention_type
# At each checkpoint, models params are saved, model is evaluated, and results printed
steps_per_checkpoint = 300

glove_dir = 'data/dwr'
glove_possible_dimensions = [50,100,200,300]
learning_rate = 0.001 # adam,   0.5 for sgd
learning_rate_decay_factor = 0.99
max_gradient_norm = 5.0
dropout_keep = 0.7

if not debug:
    """ Buckets: a list of pairs (I, O), where I specifies maximum input length
        that will be processed in that bucket, and O specifies maximum output
        length. Training instances that have inputs longer than I or outputs
        longer than O will be pushed to the next bucket and padded accordingly.
        We assume that the list is sorted, e.g., [(2, 4), (8, 16)].
    """
    _buckets = [(8, 9), (15, 16), (28, 29), (45, 46), (59,61),(100,100)]
    # These are maximums
    max_vocabulary_size = 20000
    if useMovieData:
        trainMovieQ = 'data/allMovieData.enc'
        trainMovieA = 'data/allMovieData.dec'
        reduced_weight = 0.5
    else:
        trainMovieQ = None
        trainMovieA = None
        reduced_weight = 1.0
    layer_size = 1024
    # Samples for sampled softmax.
    num_samples = 512
    batch_size = 64
    glove_dim = glove_possible_dimensions[3]
    num_layers = 3          # Number of RNN layers.
    # folder where checkpoints, vocabulary, temporary data will be stored
    working_directory = 'working_dir_%s_%s' % (glove_dim, attention_type)
else:
    _buckets = [(5, 11), (6, 16)]
    # These are maximums
    max_vocabulary_size = 10000
    trainMovieQ = None
    trainMovieA = None
    layer_size = 65
    # Samples for sampled softmax.
    num_samples = 64
    batch_size = 32
    reduced_weight = 1.0
    glove_dim = glove_possible_dimensions[0]
    num_layers = 1  # Number of RNN layers.
    # folder where checkpoints, vocabulary, temporary data will be stored
    working_directory = 'working_dir_debug_%s_%s' % (glove_dim, attention_type)

max_sentence_word_count = max([b for bucket in _buckets for b in bucket])


# Filenames.
dataDir = 'data'
autogenVocabDir = 'autogenVocabDir_%s' % (glove_dim)
# make sure these directories exist.
if not os.path.exists(dataDir):
    print "This directory must exist."
    sys.exit()
if not os.path.exists(autogenVocabDir):
    os.makedirs(autogenVocabDir)
if not os.path.exists(working_directory):
    os.makedirs(working_directory)
trainQ = 'trainQuestions.txt'
trainA = 'trainAnswers.txt'
devQ = 'validationQuestions.txt'
devA = 'validationAnswers.txt'
vocabFileName = 'vocabAll.txt'
train_enc = os.path.join(dataDir,trainQ)
train_dec = os.path.join(dataDir,trainA)
dev_enc = os.path.join(dataDir,devQ)
dev_dec = os.path.join(dataDir,devA)
id_file_train_enc = os.path.join(autogenVocabDir,trainQ+'_ids')
id_file_train_dec = os.path.join(autogenVocabDir,trainA+'_ids')
dev_enc_path = os.path.join(autogenVocabDir,devQ+'_ids')
dev_dec_path = os.path.join(autogenVocabDir,devA+'_ids')
vocabPath = os.path.join(autogenVocabDir, vocabFileName)

if trainMovieQ and trainMovieA:
    train_movie_enc = os.path.join(dataDir,trainMovieQ)
    id_file_train_movie_enc = os.path.join(autogenVocabDir,trainMovieQ+'_ids')
    train_movie_dec = os.path.join(dataDir,trainMovieA)
    id_file_train_movie_dec = os.path.join(autogenVocabDir,trainMovieA+'_ids')
else:
    train_movie_enc = None
    train_movie_dec = None
    id_file_train_movie_enc = None
    id_file_train_movie_dec = None

glove_word_embeddings_path =  os.path.join(autogenVocabDir,'glove_compressed_embeddings_%s' % glove_dim)


# Checking GPU.
# from tensorflow.python.client import device_lib
# def get_available_gpus():
#     local_device_protos = device_lib.list_local_devices()
#     return [x.name for x in local_device_protos if x.device_type == 'GPU']






