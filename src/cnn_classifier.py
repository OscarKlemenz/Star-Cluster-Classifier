import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import config as conf

def load_images():
    """ Loads the images which will be used for training.

    Returns:
        train, validate: Preprocessed train, test data which will be used for training
    """

    # Normalise pixel values
    train_datagen = ImageDataGenerator(rescale=1./255) 
    validation_datagen = ImageDataGenerator(rescale=1./255)

    # Load images from the directories
    train = train_datagen.flow_from_directory(
        conf.TRAIN_DIR,
        target_size=(64, 64),
        color_mode='grayscale',
        batch_size=32,  # Number of images per batch
        class_mode='categorical'
    )

    validate = validation_datagen.flow_from_directory(
        conf.VALIDATION_DIR,
        target_size=(64, 64),
        color_mode='grayscale', 
        batch_size=32,
        class_mode='categorical'
    )

    return train, validate


def create_model():
    """ Creates the CNN model which will be used for training. 

    Returns:
        model : The created CNN model
    """

    model = models.Sequential()

    # Example list of layers, COULD HAVE THIS IN CONFIGURATION
    layer_list = [
        layers.Conv2D(32, (3, 3), activation='relu'),  # Conv Layer 1
        layers.MaxPooling2D((2, 2)),  # Max Pooling 1
        layers.Conv2D(64, (3, 3), activation='relu'),  # Conv Layer 2
        layers.MaxPooling2D((2, 2)),  # Max Pooling 2
        layers.Conv2D(64, (3, 3), activation='relu'),  # Conv Layer 3
        layers.Flatten(),  # Flatten for Dense Layers
        layers.Dense(64, activation='relu')  # Fully Connected Layer
    ]

    model.add(layers.InputLayer(input_shape=(32, 32, 3)))

        # Add layers from the list
    for layer in layer_list:
        model.add(layer)
    
    # Add output layer for classification
    model.add(layers.Dense(2, activation='softmax'))  # Final layer
    
    # Compile the model
    model.compile(optimizer='adam',  # Optimizer
                  loss='categorical_crossentropy',  # Loss function for classification
                  metrics=['accuracy'])  # Metric to track
    
    return model