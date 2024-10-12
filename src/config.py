'''
Centralised location for model, training and dataset parameters
'''

# Data configurations
DATA_DIR = './data/'
IMAGE_SIZE = 64
IMAGE_CHANNELS = 2

# CSV Processing
# The Lowest M_V value where the image definitely looks like stars
NEGATIVE_END = -2.728
# The Start M_V where the image definitely looks like a cluster
POSITIVE_START = -5

# Filenames
SYNTH_START = 'cluster_'
SYNTH_END = '_B.fits'


# Training hyperparameters
LEARNING_RATE = 0.0001
DROPOUT = 0.5