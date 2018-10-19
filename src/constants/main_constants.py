LOG_FILE = 'LogFile-{%}.log'
BATCH_SIZE = 64

# Glove and Embedding Constants
GLOVE_TOKENS = 42  # Billion
EMBEDDING_SIZE = 300  # Dimensions
GLOVE = '../glove/glove.{}B.{}d.txt'.format(GLOVE_TOKENS, EMBEDDING_SIZE)
GLOVE_SAVE = '../glove/loaded.pkl'
TOKEN_TO_IDX_SAVE = '../data/token_to_index.pkl'
TOKEN_WEIGHTS_SAVE = '../data/token_weights.pkl'
UNK_TOKEN = '<UNK>'
BEG_TOKEN = '<BEG'
END_TOKEN = '<END>'
UNK_IDX = 0
BEG_IDX = 1
END_IDX = 2

# Data Constants
DATA_DIR = '../data/'

# Debug Constants
DEBUG_DATA_SIZE = 1000
DEBUG_BATCH_SIZE = 16

# Model Constants
AGG_EMB_SAVE_MODEL = '../save/agg_emb_model_accuracy_{:.2f}'
AGG_SAVE_MODEL = '../save/agg_model_accuracy_{:.2f}'
# Aggregate Predictor Parameters
AGG_GRAD_CLIP = 0.1
AGG_CNN_NUM_FILTERS = EMBEDDING_SIZE  # Number of filters should be equal to embedding size
AGG_CNN_KERNEL_SIZE = (3, EMBEDDING_SIZE)  # Kernel Width should be equal to embedding size
AGG_CNN_STRIDE = 1

AGG_RNN_LAYERS = 2
AGG_RNN_SIZE = 128

AGG_CNN_DROPOUT = .3
AGG_RNN_DROPOUT = .7

