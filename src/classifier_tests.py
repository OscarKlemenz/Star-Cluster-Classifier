from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt

def visualize_feature_maps(model, image):
    """ Checks the feature map when inputting one image (must be 64,64 and grayscale)

    Args:
        model: Trained model
        image: Image to test on
    """
    # Create a new model that outputs intermediate layers
    layer_outputs = [layer.output for layer in model.layers if 'conv' in layer.name]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)

    # Prepare the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    activations = activation_model.predict(image)

    # Visualize each feature map
    for layer_idx, layer_activation in enumerate(activations):
        num_filters = layer_activation.shape[-1]
        size = layer_activation.shape[1]

        # Display the feature maps
        fig, axes = plt.subplots(1, num_filters, figsize=(20, 20))
        fig.suptitle(f'Feature Maps for Layer {layer_idx + 1}')

        for i in range(num_filters):
            ax = axes[i]
            ax.matshow(layer_activation[0, :, :, i], cmap='viridis')
            ax.axis('off')
        plt.show()
