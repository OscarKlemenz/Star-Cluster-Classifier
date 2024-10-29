from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import config as conf
import os

def visualize_feature_maps(model, image_path):
    """ Checks the feature map when inputting one image (must be 64x64 and grayscale)

    Args:
        model: Trained model
        image_path: Path to the image file to test on
    """
    
    # Compile the model if it wasn't compiled after loading
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Make a dummy prediction to initialize the model
    dummy_image = np.zeros((1, 64, 64, 1))  # Batch size of 1, 64x64, 1 channel (grayscale)
    model.predict(dummy_image)  # This will initialize the model

    # Create a new model that outputs intermediate layers
    layer_outputs = [layer.output for layer in model.layers if 'conv' in layer.name]
    
    if not layer_outputs:
        print("No convolutional layers found in the model.")
        return

    activation_model = Model(inputs=model.input, outputs=layer_outputs)

    # Load and preprocess the image
    image = load_img(image_path, target_size=(64, 64), color_mode='grayscale')  # Load image
    image = img_to_array(image) / 255.0  # Convert to array and normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Get activations
    activations = activation_model.predict(image)

    # Visualize each feature map
    for layer_idx, layer_activation in enumerate(activations):
        num_filters = layer_activation.shape[-1]

        # Display the feature maps
        fig, axes = plt.subplots(1, num_filters, figsize=(20, 20))
        fig.suptitle(f'Feature Maps for Layer {layer_idx + 1}')

        for i in range(num_filters):
            ax = axes[i]
            ax.matshow(layer_activation[0, :, :, i], cmap='viridis')
            ax.axis('off')
        plt.show()

def predict_image(model, image_path):
    """ Takes an image path, preprocesses the image, and predicts the class.

    Args:
        model: The trained model to make predictions with.
        image_path: Path to the image file.

    Returns:
        The predicted class label and the associated probability.
    """
    # Load and preprocess the image
    img = load_img(image_path, target_size=(conf.IMAGE_SIZE, conf.IMAGE_SIZE), color_mode='grayscale')
    img_array = img_to_array(img) / 255.0  # Normalize the pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make predictions
    predictions = model.predict(img_array)
    
    # Get the predicted class
    predicted_class = np.argmax(predictions, axis=1)
    predicted_prob = np.max(predictions)  # Get the maximum probability

    return predicted_class[0], predicted_prob

# Function to load and preprocess images
def load_and_preprocess_image(image_path, target_size=(64, 64)):
    """Loads and preprocesses an image."""
    img = load_img(image_path, target_size=target_size, color_mode='grayscale')
    img_array = img_to_array(img) / 255.0  # Normalize the pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def predict_images_in_directory(model, directory_path):
    """Predicts classes for all images in the specified directory."""
    results = []

    # Iterate over each subdirectory in the test directory
    for class_folder in os.listdir(directory_path):
        class_folder_path = os.path.join(directory_path, class_folder)

        # Check if it is a directory
        if os.path.isdir(class_folder_path):
            # Iterate over each file in the class folder
            for filename in os.listdir(class_folder_path):
                # Construct full file path
                file_path = os.path.join(class_folder_path, filename)

                # Check if the file is an image
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    # Load and preprocess the image
                    img_array = load_and_preprocess_image(file_path)

                    # Make predictions
                    predictions = model.predict(img_array)
                    predicted_class = np.argmax(predictions, axis=1)[0]  # Get predicted class index
                    predicted_prob = np.max(predictions)  # Get the maximum probability

                    # Store the result with the actual label based on the folder name
                    results.append({
                        'filename': filename,
                        'actual_class': class_folder,  # The folder name represents the actual class
                        'predicted_class': predicted_class,
                        'probability': predicted_prob
                    })

    return results

if __name__ == "__main__":
    # Load the trained model
    model = tf.keras.models.load_model("./models/TEST.h5")

    # Directory containing test images
    test_directory = "./data/dataset/test"  # Replace with your test images directory

    # Make predictions on all images in the test directory
    # predictions = predict_images_in_directory(model, test_directory)

    # # Display the results
    # for result in predictions:
    #      print(f"File: {result['filename']}, Predicted Class: {result['predicted_class']}, Probability: {result['probability']:.2f}")

    print(predict_image(model, "./data/dataset/test/non-cluster/cluster_0001_B_aug_11.png"))