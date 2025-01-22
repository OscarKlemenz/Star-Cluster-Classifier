import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import config as conf
import random
import shutil
import os

def fits_to_png(fits_file, output_png, image_size=64):
    """
    Convert a FITS file to PNG and resize it to a specific target size.
    
    Parameters:
    - fits_file: path to the input FITS file.
    - output_png: path to save the PNG file.
    - target_size: representing the desired image size, e.g., 64.
    """
    
    # Open the FITS file
    with fits.open(fits_file) as hdul:
        data = hdul[0].data  # Assuming the image is in the primary HDU
    
    # Replace NaNs or Infs with zeros to avoid issues
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    # Apply Zscale normalization to raw data
    zscale = ZScaleInterval()
    vmin, vmax = zscale.get_limits(data)

    # Clip the data to the Zscale range
    clipped_data = np.clip(data, vmin, vmax)

    # Apply asinh scaling on the clipped data
    scaled_data = np.arcsinh(clipped_data)

    # Recompute the min/max for the asinh-transformed data
    scaled_vmin = scaled_data.min()
    scaled_vmax = scaled_data.max()

    # Normalize the scaled data to the range [0, 255] for 8-bit image representation
    scaled_data = (scaled_data - scaled_vmin) / (scaled_vmax - scaled_vmin)
    scaled_data = (scaled_data * 255).astype(np.uint8)

    # Create an image using PIL
    img = Image.fromarray(scaled_data)

    # Resize the image if a specific size is provided
    if image_size is not None:
        img = img.resize((image_size, image_size), Image.LANCZOS)

    # Save the image as PNG
    img.save(output_png)

def convert_fits_from_directory(source_folder, output_folder, image_size=(64, 64)):
    """
    Convert all FITS files in the specified directory to PNG and save them in the output folder.
    
    Parameters:
    - source_folder: Directory containing the input FITS files.
    - output_folder: Folder where the PNG files will be saved.
    - image_size: Tuple representing the desired image size, e.g., (64, 64).
    """
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over each file in the source directory
    for filename in os.listdir(source_folder):
        if filename.endswith('.fits'):
            fits_file = os.path.join(source_folder, filename)
            output_png = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.png")
            fits_to_png(fits_file, output_png, image_size)

def collect_samples_by_mv(csv_filename):
    """Gets the negative and positive sample names from a csv file
    based upon their M_V values

    Args:
        csv_filename (str): Name of the csv to extract the names
    Returns:
        str[], str[]: List of ids for both positive and negative samples
    """
    df = pd.read_csv(csv_filename)

    negative_ids = []
    positive_ids = []

    for _, row in df.iterrows():
        
        if row['M_V'] > conf.NEGATIVE_END:
            negative_ids.append(str(row['ID']).zfill(4))
        elif row['M_V'] < conf.POSITIVE_START:
            positive_ids.append(str(row['ID']).zfill(4))
    
    return positive_ids, negative_ids

def generate_filenames(ids):
    """ Adds the correct filename format onto collected ids

    Args:
        ids (str[]): List of ids

    Returns:
        str[]: List of filenames
    """

    for i in range(0, len(ids)):
        ids[i] = conf.SYNTH_START + ids[i] + conf.SYNTH_END
    
    return ids

def place_data_in_new_folder(source_folder, dest_folder, filenames):
    """ Moves files from one folder to another

    Args:
        source_folder (str): Folder the files are coming from
        dest_folder (str): Folder the files are arriving in
        filenames (str[]): List of filenames ot move
    """
    
    if not os.path.exists(source_folder):
        print(f"Source folder '{source_folder}' does not exist.")
        return

    if not os.path.exists(dest_folder):
        # Create the destination folder if it doesn't exist
        os.makedirs(dest_folder)

    # Loop through all files in the source folder
    for filename in os.listdir(source_folder):
        if filename in filenames:
            # Get the full file path
            full_file_name = os.path.join(source_folder, filename)
            
            # Check if it's a file (not a subdirectory)
            if os.path.isfile(full_file_name):
                # Copy the file to the destination folder
                shutil.copy(full_file_name, dest_folder)
                print(f"Copied: {filename}")
    
    print("File copying completed.")

def data_augmentation(original_images_dir, augmented_images_dir, no_of_augmentations=2):
    """Applies data augmentation to a directory of images

    Args:
        original_images_dir (str): Source of the images
        augmented_images_dir (str): Destination for the new augmented images
    """

    # Background grey value
    #background_grey_value = 94  # Set to the most common pixel value
    
    # Augmentation properties
    datagen = ImageDataGenerator(
        width_shift_range=0.2,    # Horizontal shift (10% of the image width)
        height_shift_range=0.2,   # Vertical shift (10% of the image height)
        rotation_range=15,        # Rotate images by up to 15 degrees
        zoom_range=0.1,           # Random zoom in/out by 10%
        horizontal_flip=True,     # Randomly flip images horizontally
        vertical_flip=False,      # Optionally flip images vertically
        fill_mode='wrap'          # Shift without stretching the image
    )

    # Create the output directory if it doesn't exist
    if not os.path.exists(augmented_images_dir):
        os.makedirs(augmented_images_dir)

    # Load and augment images
    for filename in os.listdir(original_images_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(original_images_dir, filename)
            
            # Save original to augmented folder
            shutil.copy(img_path, os.path.join(augmented_images_dir, filename))

            # Load the image in grayscale
            img = load_img(img_path, color_mode='grayscale')  # Load as grayscale
            img_array = img_to_array(img)  # Convert to numpy array
            img_array = img_array.reshape((1,) + img_array.shape)  # Reshape for the generator
            
            # Get the base filename (without extension) for saving augmented images
            base_filename = os.path.splitext(filename)[0]  # Removes the file extension
            # Augment the image but do not save directly
            i = 0
            for batch in datagen.flow(img_array, batch_size=1):
                
                # Convert the batch (augmented image) to array for post-processing
                aug_img = batch[0].astype(np.uint8)

                # Replace black pixels (0) with the background grey value (94)
                #aug_img[aug_img == 0] = background_grey_value

                # Convert back to an image and save in grayscale mode
                result_img = Image.fromarray(aug_img.squeeze(), mode='L')  # 'L' mode for grayscale
                result_img.save(f'{augmented_images_dir}/{base_filename}_aug_{i}.png')
                
                i += 1
                if i >= no_of_augmentations:  # Specifies the amount of augmentation per image
                    break

def split_data(source_dir, dest_dir):
    """Splits images in the source directory into train, test, and validation sets 
    and organizes them into class subdirectories.

    Args:
        source_dir (str): Directory containing the original images (organized by class).
        dest_dir (str): Directory to save the split datasets (train, test, validate).
        train_ratio (float): Proportion of the dataset to be used for training.
        test_ratio (float): Proportion of the dataset to be used for testing.
        val_ratio (float): Proportion of the dataset to be used for validation.

    The sum of `train_ratio`, `test_ratio`, and `val_ratio` should be 1.
    """
    # Ensure the ratios sum to 1
    if conf.TRAIN_RATIO + conf.TEST_RATIO + conf.VAL_RATIO != 1.0:
        raise ValueError("Train, test, and validation ratios must sum to 1.")

    # Create destination directories if they don't exist
    train_dir = os.path.join(dest_dir, 'train')
    test_dir = os.path.join(dest_dir, 'test')
    val_dir = os.path.join(dest_dir, 'validate')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Iterate over each class folder in the source directory
    for class_name in os.listdir(source_dir):
        class_dir = os.path.join(source_dir, class_name)
        
        if not os.path.isdir(class_dir):
            continue  # Skip if it's not a directory

        # Get all image files in the class directory
        all_files = [f for f in os.listdir(class_dir) if f.endswith('.jpg') or f.endswith('.png')]
        
        # Shuffle the files randomly
        random.seed(42)  # For reproducibility
        random.shuffle(all_files)

        # Calculate the number of files for each split
        total_files = len(all_files)
        train_count = int(total_files * conf.TRAIN_RATIO)
        test_count = int(total_files * conf.TEST_RATIO)
        val_count = total_files - train_count - test_count  # Remaining files for validation

        # Split the files into train, test, and validation sets
        train_files = all_files[:train_count]
        test_files = all_files[train_count:train_count + test_count]
        val_files = all_files[train_count + test_count:]

        # Function to copy files to their respective directories, maintaining class folders
        def copy_files(file_list, target_dir):
            class_target_dir = os.path.join(target_dir, class_name)  # Create class subfolder
            os.makedirs(class_target_dir, exist_ok=True)  # Create class folder if it doesn't exist

            for file_name in file_list:
                src_path = os.path.join(class_dir, file_name)
                dst_path = os.path.join(class_target_dir, file_name)
                shutil.copy(src_path, dst_path)

        # Copy the files into the respective directories
        copy_files(train_files, train_dir)
        copy_files(test_files, test_dir)
        copy_files(val_files, val_dir)

        print(f"Class '{class_name}' split into train, test, and validate sets.")
        print(f"  Training set: {len(train_files)} images")
        print(f"  Test set: {len(test_files)} images")
        print(f"  Validation set: {len(val_files)} images")

    print("Data split complete.")

def resize_images_in_folder(source_folder, target_folder, target_size=(64, 64)):
    """Resizes all images in the source folder to the target size and saves them in the target folder.

    Args:
        source_folder (str): Path to the folder containing the original images.
        target_folder (str): Path to the folder to save the resized images.
        target_size (tuple): Desired image size, e.g., (64, 64).
    """
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for filename in os.listdir(source_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(source_folder, filename)
            img = Image.open(img_path)
            img_resized = img.resize(target_size, Image.LANCZOS)
            img_resized.save(os.path.join(target_folder, filename))
            print(f"Resized and saved {filename}")

if __name__ == "__main__":
    
    # # Get positive and negative samples
    # pos, neg = collect_samples_by_mv('./data/csv/synthetic_clusters_ordby_M_V.csv')
    # # # # Create filenames
    # pos = generate_filenames(pos)
    # neg = generate_filenames(neg)
    # print('Located Cluster and Non-Cluster Files')
    # # Create new folders
    # place_data_in_new_folder(conf.SYNTH_SOURCE, conf.SYNTH_DEST_POS, pos)
    # place_data_in_new_folder(conf.SYNTH_SOURCE, conf.SYNTH_DEST_NEG, neg)
    # print('Moved Files')
    # # # Convert the files to images
    # convert_fits_from_directory(conf.SYNTH_DEST_POS , conf.SYNTH_DEST_POS + '_png', conf.IMAGE_SIZE)
    # convert_fits_from_directory(conf.SYNTH_DEST_NEG , conf.SYNTH_DEST_NEG + '_png', conf.IMAGE_SIZE)
    # print('Converted Fits files to png')
    # Resize the images
    # resize_images_in_folder("./data/real_nonclean_negative", "./data/real_nonclean_positive_resized")
    # resize_images_in_folder("./data/Yilun_Wang_cutouts", "./data/Yilun_Wang_cutouts_resized")
    # Augment the data
    # data_augmentation('./data/pre-split_128/cluster', './data/pre-split_128/cluster_aug', 6)
    # data_augmentation('./data/pre-split_128/non-cluster', './data/pre-split_128/non-cluster_aug', 20)
    #print('Augmented Data')
    # Split the data
    split_data('./data/pre-split_128', './data/dataset_128')
    print('Split Data')
    # fits_to_png('./data/synthetic_clusters/cluster_0032_B.fits', './data/test_images/cluster_0032_B_diss.png', 128)
