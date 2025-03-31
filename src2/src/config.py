'''
Centralised location for model, training and dataset parameters
'''

# Data configurations
DATA_DIR = './data/'

# CSV Processing
# The Lowest M_V value where the image definitely looks like stars
NEGATIVE_END = -2.728
# The Start M_V where the image definitely looks like a cluster
POSITIVE_START = -5

# Filenames
SYNTH_START = 'cluster_'
SYNTH_END = '_B.fits'
SYNTH_SOURCE = './data/synthetic_clusters'
SYNTH_DEST_POS = './data/synthetic_positive_128'
SYNTH_DEST_NEG = './data/synthetic_negative_128'

# Preprocessing variables
TRAIN_RATIO=0.7
TEST_RATIO=0.15
VAL_RATIO=0.15

"""
====================
CNN Classifier Config
====================
"""

# Dataset location
TRAIN_DIR = '../data/images/FINAL/SR/train'
VALIDATION_DIR = '../data/images/FINAL/SR/validate'

# Model save location
MODEL_SAVE_PATH = '../models/64SR_septest.h5'

# Training hyperparameters
LEARNING_RATE = 0.001
DROPOUT_BOOL = False
DROPOUT = 0.5
EPOCHS = 5
BATCH_SIZE = 32
IMAGE_SIZE = 64
IMAGE_CHANNELS = 1