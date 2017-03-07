# my config file.
import os

# START IMPORTANT
debug = False
attention_types = ['vinyals', 'luong', 'bahdanau']
attention_type = attention_types[1]
possibleModes = ['train', 'test', 'serve']
mode = possibleModes[0]

# TODO: Add ability for this.
useTensorBoard = False 
# command is: tensorboard --logdir=run1:/tmp/tensorflow/trump1_luong --port 6006
# useful: tensorboard --inspect --logdir=/tmp/tensorflow/trump3_vinyals
logs_path = '/tmp/tensorflow/trump3_%s' % attention_type
# END IMPORTANT

# dataset size limit; typically none : no limit
max_train_data_size = 0

# steps per checkpoint
#     Note : At a checkpoint, models parameters are saved, model is evaluated
#            and results are printed
steps_per_checkpoint = 150

# Glove.
glove_dir = 'data/dwr'
glove_possible_dimensions = [50,100,200,300]

learning_rate = 0.5
learning_rate_decay_factor = 0.99
max_gradient_norm = 5.0
dropout_keep = 0.7

if not debug:
    # Actual.
    _buckets = [(8, 9), (15, 16), (28, 29), (45, 46), (59,61)]
    # These are maximums
    max_vocabulary_size = 20000
    if False:
        trainMovieQ = 'data/allMovieData.enc'
        trainMovieA = 'data/allMovieData.dec'
        reduced_movie_weight = 0.2
    else:
        trainMovieQ = None
        trainMovieA = None
        reduced_movie_weight = None
    layer_size = 512
    # Samples for sampled softmax.
    num_samples = 256
    batch_size = 64
    glove_dim = glove_possible_dimensions[1]
    # Number of RNN layers.
    num_layers = 2
    # folder where checkpoints, vocabulary, temporary data will be stored
    working_directory = 'working_dir_%s_%s' % (glove_dim, attention_type)
else:
    # When debugging, use this.
    _buckets = [(5, 11), (6, 16)]
    # These are maximums
    max_vocabulary_size = 10000
    trainMovieQ = None
    trainMovieA = None
    layer_size = 64
    # Samples for sampled softmax.
    num_samples = 64
    batch_size = 32
    reduced_movie_weight = None
    glove_dim = glove_possible_dimensions[0]
    # Number of RNN layers.
    num_layers = 1
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
id_file_dev_enc = os.path.join(autogenVocabDir,devQ+'_ids')
id_file_dev_dec = os.path.join(autogenVocabDir,devA+'_ids')
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






