# my config file.

# START IMPORTANT
debug = True
attention_types = ['vinyals', 'luong', 'bahdanau']
attention_type = attention_types[1]
possibleModes = ['train', 'test', 'serve']
mode = possibleModes[0]

# TODO: Add ability for this.
useTensorBoard = False

logs_path = '/tmp/tensorflow/trump1_%s' % attention_type
# END IMPORTANT

train_enc = 'data/trainQuestions.txt'
train_dec = 'data/trainAnswers.txt'
dev_enc = 'data/validationAnswers.txt'
dev_dec = 'data/validationQuestions.txt'
# folder where checkpoints, vocabulary, temporary data will be stored
working_directory = 'working_dir'


# TODO: fix BUG WITH attention mechanisms for layers > 1.
# number of LSTM layers : 1/2/3
num_layers = 1
#num_layers = 3

# dataset size limit; typically none : no limit
max_train_data_size = 0

# steps per checkpoint
# 	Note : At a checkpoint, models parameters are saved, model is evaluated
#			and results are printed
steps_per_checkpoint = 150

# Samples for sampled softmax.

learning_rate = 0.5
learning_rate_decay_factor = 0.99
max_gradient_norm = 5.0

if not debug:
	# Actual.
	_buckets = [(5, 11), (6, 16), (19, 26), (39, 51), (59,61)]
	# These are maximums
	enc_vocabulary_size = 20000
	dec_vocabulary_size = 20000
	train_movie_enc = 'data/allMovieData.enc'
	train_movie_dec = 'data/allMovieData.dec'
	layer_size = 512
	num_samples = 256
	batch_size = 64
	reduced_movie_weight = 0.2
else:
	# When debugging, use this.
	_buckets = [(5, 11), (6, 16)]
	# These are maximums
	enc_vocabulary_size = 10000
	dec_vocabulary_size = 10000
	train_movie_enc = None
	train_movie_dec = None
	layer_size = 64
	num_samples = 64
	batch_size = 32
	reduced_movie_weight = None

max_sentence_word_count = max([b for bucket in _buckets for b in bucket])




# Checking GPU.
# from tensorflow.python.client import device_lib
# def get_available_gpus():
#     local_device_protos = device_lib.list_local_devices()
#     return [x.name for x in local_device_protos if x.device_type == 'GPU']






