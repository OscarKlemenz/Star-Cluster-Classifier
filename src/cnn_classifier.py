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
        target_size=(conf.IMAGE_SIZE, conf.IMAGE_SIZE),
        color_mode='grayscale',
        batch_size=32,
        class_mode='categorical'
    )

    validate = validation_datagen.flow_from_directory(
        conf.VALIDATION_DIR,
        target_size=(conf.IMAGE_SIZE, conf.IMAGE_SIZE),
        color_mode='grayscale', 
        batch_size=32,
        class_mode='categorical'
    )

    return train, validate


def create_model():
    """Creates the CNN model based on the provided architecture.
    
    Returns:
        model : The created CNN model
    """
    
    model = models.Sequential()
    
    # Input layer
    model.add(layers.InputLayer(input_shape=(conf.IMAGE_SIZE, conf.IMAGE_SIZE, conf.IMAGE_CHANNELS)))
    
    # Model architecture
    layer_list = [
        layers.Conv2D(32, (3, 3), activation='relu'),  # Conv Layer 1
        layers.MaxPooling2D((2, 2)),  # Max Pooling 1
        layers.Conv2D(64, (3, 3), activation='relu'),  # Conv Layer 2
        layers.MaxPooling2D((2, 2)),  # Max Pooling 2
        layers.Conv2D(64, (3, 3), activation='relu'),  # Conv Layer 3
        layers.Flatten(),
        layers.Dense(64, activation='relu')
    ]
    
    # Add layers from the list
    for layer in layer_list:
        model.add(layer)
    
    # Output layer
    model.add(layers.Dense(2, activation='softmax'))
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

if __name__ == "__main__":

    train_data, validation_data = load_images()
    model = create_model()

    history = model.fit(
        train_data,
        epochs=conf.EPOCHS,
        validation_data=validation_data
    )

    model.save("./models/128SRYC_NNS.h5")
