# Property: embedding_size:
# Embedding vector size
EMBEDDING_SIZE = 300
# Property: "diff_vocab":
# Whether the embedding matrix should be different for the encoder and decoder symbols.
DIFF_VOCAB = False
# Property: embedding_path:
# Path to the pretrained embedding
EMBEDDING_PATH = "../Embedding/embeddings"
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
