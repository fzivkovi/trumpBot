# my config file.

# START IMPORTANT
debug = True
attention_types = ['vinyals', 'luong', 'bahdanau']
attention_type = attention_types[0]
possibleModes = ['train', 'test', 'serve']
mode = possibleModes[1]

# TODO: Add ability for this.
useTensorBoard = True
# command is: tensorboard --logdir=run1:/tmp/tensorflow/trump1_luong --port 6006
logs_path = '/tmp/tensorflow/trump2_%s' % attention_type
# END IMPORTANT

train_enc = 'data/trainQuestions.txt'
train_dec = 'data/trainAnswers.txt'
dev_enc = 'data/validationAnswers.txt'
dev_dec = 'data/validationQuestions.txt'

# dataset size limit; typically none : no limit
max_train_data_size = 0

# steps per checkpoint
# 	Note : At a checkpoint, models parameters are saved, model is evaluated
#			and results are printed
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
	_buckets = [(5, 11), (6, 16), (19, 26), (39, 51), (59,61)]
	# These are maximums
	max_vocabulary_size = 20000
	if False:
		train_movie_enc = 'data/allMovieData.enc'
		train_movie_dec = 'data/allMovieData.dec'
		reduced_movie_weight = 0.2
	else:
		train_movie_enc = None
		train_movie_dec = None
		reduced_movie_weight = None
	layer_size = 512
	# Samples for sampled softmax.
	num_samples = 256
	batch_size = 64
	glove_dim = glove_possible_dimensions[3]
	save_path = 'data/dwr/gloveWordEmbeddings_%s' % glove_dim
	# Number of RNN layers.
	num_layers = 3
	# folder where checkpoints, vocabulary, temporary data will be stored
	working_directory = 'working_dir'
else:
	# When debugging, use this.
	_buckets = [(5, 11), (6, 16)]
	# These are maximums
	max_vocabulary_size = 10000
	train_movie_enc = None
	train_movie_dec = None
	layer_size = 64
	# Samples for sampled softmax.
	num_samples = 64
	batch_size = 32
	reduced_movie_weight = None
	glove_dim = glove_possible_dimensions[0]
	save_path = 'data/dwr/gloveWordEmbeddings_%s' % glove_dim
	# Number of RNN layers.
	num_layers = 1
	# folder where checkpoints, vocabulary, temporary data will be stored
	working_directory = 'working_dir_debug'

max_sentence_word_count = max([b for bucket in _buckets for b in bucket])



# Checking GPU.
# from tensorflow.python.client import device_lib
# def get_available_gpus():
#     local_device_protos = device_lib.list_local_devices()
#     return [x.name for x in local_device_protos if x.device_type == 'GPU']






