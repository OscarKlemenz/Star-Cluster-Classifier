from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import config as conf
import os
import json
import shutil
import re

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

def evaluate_model(model, test_directory):
    """Evaluates the model on the test dataset and plots accuracy, loss, precision, recall, and F1-score.

    Args:
        model: The trained model to evaluate.
        test_directory: Path to the directory containing test images.
    """
    # Load test data
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_data = test_datagen.flow_from_directory(
        test_directory,
        target_size=(conf.IMAGE_SIZE, conf.IMAGE_SIZE),
        color_mode='grayscale',
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )

    # Evaluate the model
    scores = model.evaluate(test_data)
    print(f"Test Loss: {scores[0]}")
    print(f"Test Accuracy: {scores[1]}")

    # Predict the classes
    y_pred = model.predict(test_data)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_data.classes

    # Calculate precision, recall, and F1-score manually
    precision = np.zeros(len(test_data.class_indices))
    recall = np.zeros(len(test_data.class_indices))
    f1_score = np.zeros(len(test_data.class_indices))

    for i in range(len(test_data.class_indices)):
        true_positives = np.sum((y_true == i) & (y_pred_classes == i))
        false_positives = np.sum((y_true != i) & (y_pred_classes == i))
        false_negatives = np.sum((y_true == i) & (y_pred_classes != i))

        precision[i] = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall[i] = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0

    # Print precision, recall, and F1-score
    for i, class_name in enumerate(test_data.class_indices.keys()):
        print(f"Class: {class_name}")
        print(f"Precision: {precision[i]}")
        print(f"Recall: {recall[i]}")
        print(f"F1-score: {f1_score[i]}")
        print()

    # Plot confusion matrix
    conf_matrix = np.zeros((len(test_data.class_indices), len(test_data.class_indices)), dtype=int)
    for true, pred in zip(y_true, y_pred_classes):
        conf_matrix[true, pred] += 1

    plt.figure(figsize=(10, 7))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(test_data.class_indices))
    plt.xticks(tick_marks, test_data.class_indices.keys(), rotation=45)
    plt.yticks(tick_marks, test_data.class_indices.keys())

    for i in range(len(test_data.class_indices)):
        for j in range(len(test_data.class_indices)):
            plt.text(j, i, conf_matrix[i, j], horizontalalignment="center", color="white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    # Plot accuracy and loss
    history = model.history.history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Model Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Model Loss')

    plt.show()

def classify_and_organize_images(model, source_directory, output_directory):
    """Classifies images in the source directory and organizes them into 'cluster' and 'non-cluster' folders.

    Args:
        model: The trained model to make predictions with.
        source_directory: Path to the directory containing images to classify.
        output_directory: Path to the directory where classified images will be saved.
    """
    # Create output directories if they don't exist
    cluster_dir = os.path.join(output_directory, 'cluster')
    non_cluster_dir = os.path.join(output_directory, 'non-cluster')
    os.makedirs(cluster_dir, exist_ok=True)
    os.makedirs(non_cluster_dir, exist_ok=True)

    # Iterate over each file in the source directory
    for filename in os.listdir(source_directory):
        file_path = os.path.join(source_directory, filename)

        # Check if the file is an image
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            # Load and preprocess the image
            img_array = load_and_preprocess_image(file_path, target_size=(conf.IMAGE_SIZE, conf.IMAGE_SIZE))

            # Make predictions
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=1)[0]  # Get predicted class index

            # Determine the target directory based on the predicted class
            target_dir = cluster_dir if predicted_class == 0 else non_cluster_dir

            # Copy the image to the target directory
            shutil.copy(file_path, target_dir)

def plot_predictions(model, test_directory):
    """Makes predictions on the test dataset and plots RA and DEC coordinates.

    Args:
        model: The trained model to make predictions with.
        test_directory: Path to the directory containing test images.
    """
    # Lists to store RA and DEC values for plotting
    ra_correct = []
    dec_correct = []
    ra_incorrect = []
    dec_incorrect = []

    # Function to extract RA and DEC from filename
    def extract_ra_dec(filename):
        match = re.search(r'_(\d+\.\d+)_(\d+\.\d+)\.png$', filename)
        if match:
            ra = float(match.group(1))
            dec = float(match.group(2))
            return ra, dec
        return None, None

    # Iterate over each subdirectory in the test directory
    for class_folder in os.listdir(test_directory):
        class_folder_path = os.path.join(test_directory, class_folder)

        # Check if it is a directory
        if os.path.isdir(class_folder_path):
            # Iterate over each file in the class folder
            for filename in os.listdir(class_folder_path):
                # Construct full file path
                file_path = os.path.join(class_folder_path, filename)

                # Check if the file is an image
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    # Load and preprocess the image
                    img_array = load_and_preprocess_image(file_path, target_size=(conf.IMAGE_SIZE, conf.IMAGE_SIZE))

                    # Make predictions
                    predictions = model.predict(img_array)
                    predicted_class = np.argmax(predictions, axis=1)[0]  # Get predicted class index

                    # Extract RA and DEC from the filename
                    ra, dec = extract_ra_dec(filename)

                    # Check if the prediction is correct
                    actual_class = 0 if class_folder == 'cluster' else 1
                    if predicted_class == actual_class:
                        ra_correct.append(ra)
                        dec_correct.append(dec)
                    else:
                        ra_incorrect.append(ra)
                        dec_incorrect.append(dec)

    # Create a scatter plot
    plt.figure(figsize=(10, 6))

    # Plot correct and incorrect points with different colors
    plt.scatter(ra_correct, dec_correct, color='green', label='Correct', alpha=0.6)
    plt.scatter(ra_incorrect, dec_incorrect, color='red', label='Incorrect', alpha=0.6)

    # Add labels and title
    plt.xlabel('RA (degrees)')
    plt.ylabel('DEC (degrees)')
    plt.title('RA and DEC Coordinates by Prediction Accuracy')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

if __name__ == "__main__":
    # Load the trained model
    model = tf.keras.models.load_model("./models/128SRNC.h5")

    # Directory containing images to classify
    source_directory = "./data/Yilun_Wang_Cutouts"  # Replace with your source images directory

    # Directory to save classified images
    output_directory = "./data/predictions_128SRNC"

    # Directory containing test images
    test_directory = "./data/dataset_128/test"

    # Test the models accuracy
    # evaluate_model(model, "./data/dataset_128/test")

    # Plot predictions
    plot_predictions(model, test_directory)

    # Classify and organize images
    # classify_and_organize_images(model, source_directory, output_directory)
    # # Make predictions on all images in the test directory
    # predictions = predict_images_in_directory(model, test_directory)

    # # Display the results
    # for result in predictions:
    #      print(f"File: {result['filename']}, Predicted Class: {result['predicted_class']}, Probability: {result['probability']:.2f}")

    #print(predict_image(model, "./data/dataset/test/non-cluster/cluster_0001_B_aug_11.png"))