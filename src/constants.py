# Property: embedding_size:
# Embedding vector size
EMBEDDING_SIZE = 300
# Property: "diff_vocab":
# Whether the embedding matrix should be different for the encoder and decoder symbols.
DIFF_VOCAB = False
# Property: embedding_path:
# Path to the pretrained embedding
EMBEDDING_PATH = "../Embedding/"
#Property: limit_encode:
# Frequency used as a cutoff for encoder vocab
LIMIT_ENCODE = 0
#Property limit_decode:
# Frequency used as a cutoff for decoder vocab
LIMIT_DECODE = 0
# Property: working_dir:
# Path to the folder that contains train/valid/test data
WORKING_DIR = "../data/1/"
# Property: batch_size:
# Batch size for training
BATCH_SIZE = 64
# Property: num_epochs:
# Number of maximum epochs
MAX_EPOCHS = 50
# Property: Early stop:
# Number of epochs for early stop criteria	
EARLY_STOP = 5
PRINT_FREQUENCY = 100
# Property : hidden_size:
# RNN cell size
HIDDEN_SIZE = 400
# Property: learning_rate:
# Learning rate to be used for training.
LEARNING_RATE = 0.0004
#Initial Learning rate
GRAD_CLIP = 10.0
# Property: outdir:
# Path to the output directory where the model and the predictions will be saved
OUTDIR = "output/"
