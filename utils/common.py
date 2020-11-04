SEED = 7

# sequence service tokens
SOS = "<SOS>"
EOS = "<EOS>"
PAD = "<PAD>"
UNK = "<UNK>"

# path context keys
FROM_TOKEN = "from_token"
PATH_NODES = "path_nodes"
TO_TOKEN = "to_token"

# vocabulary keys
TOKEN_TO_ID = "token_to_id"
NODE_TO_ID = "node_to_id"
LABEL_TO_ID = "label_to_id"

# dataset keys
DATA_FOLDER = "data"
VOCABULARY_NAME = "vocabulary.pkl"
TRAIN_HOLDOUT = "train"
VAL_HOLDOUT = "val"
TEST_HOLDOUT = "test"
HOLDOUTS = [TRAIN_HOLDOUT, VAL_HOLDOUT, TEST_HOLDOUT]
