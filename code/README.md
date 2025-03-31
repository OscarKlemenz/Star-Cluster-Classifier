# Star-Cluster-Classifier

This is the repository of code for my work on my literature review - <b>Classification of star clusters in astronomical imaging using synthetic clusters as a training sample</b>

<i>Currently work in progress</i>

## Requirements

Before you begin, ensure you have the following installed:

- Python 3.x (preferably the latest version)
- pip (Python package installer)

## Installation

Follow these steps to set up a virtual environment and install the necessary dependencies for this project:

1. **Clone the repository** (replace `<repository-url>` with the URL of your repository):
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Create a virtual environment**:

On macOS and Linux:
```bash
python3 -m venv myenv
```
On Windows:
```bash
python -m venv myenv
```

3. **Activate the virtual environment**:

On macOS and Linux:
```bash
source myenv/bin/activate
```

On Windows:
```bash
myenv\Scripts\activate
```

4. **Install the required packages**: 

With the virtual environment activated, run:
```bash
pip install -r requirements.txt
```

### The Virtual Environment

Exiting the environment:
`deactivate`

## Project Structure

### Folders and Code Files

- **data**: This folder holds the different catalogues containing astronomical phenomena. These catalogues are cross-referenced with their corresponding FITS files in `locating_real_images.ipynb`.

- **models**: This folder contains the trained models. Due to their large sizes, models and images are found on OneDrive.

- **src/classifier_tests.ipynb**: This notebook contains all the tests run on the CNN to test its functionality.

- **src/cnn_classifier.ipynb**: This notebook is used for training CNNs.

- **src/data_preprocessing.ipynb**: This notebook contains the preprocessing pipeline for processing all images that the CNN is trained on.

- **src/visualisations.ipynb**: This notebook is used for any graphing needed for the dissertation.

- **src/heatmap.ipynb**: This notebook is used to create integrated gradient images of the CNN's predictions to understand why it has made certain decisions.