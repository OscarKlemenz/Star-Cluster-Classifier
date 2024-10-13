import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from PIL import Image
import config as conf
import shutil
import os

def fits_to_png(fits_file, output_png, image_size=(64, 64)):
    """
    Convert a FITS file to PNG and resize it to a specific target size.
    
    Parameters:
    - fits_file: path to the input FITS file.
    - output_png: path to save the PNG file.
    - target_size: tuple representing the desired image size, e.g., (64, 64).
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


if __name__ == "__main__":
    
    # # Get positive and negative samples
    # pos, neg = collect_samples_by_mv('./data/csv/synthetic_clusters_ordby_M_V.csv')
    # # Create filenames
    # pos = generate_filenames(pos)
    # neg = generate_filenames(neg)
    # # Create new folders
    # place_data_in_new_folder(conf.SYNTH_SOURCE, conf.SYNTH_DEST_POS, pos)
    # place_data_in_new_folder(conf.SYNTH_SOURCE, conf.SYNTH_DEST_NEG, neg)
    # Convert the files to images
    convert_fits_from_directory(conf.SYNTH_DEST_POS , conf.SYNTH_DEST_POS + '_png', conf.IMAGE_SIZE)
    convert_fits_from_directory(conf.SYNTH_DEST_NEG , conf.SYNTH_DEST_NEG + '_png', conf.IMAGE_SIZE)

    # Next perform data augmentation
    

# fits_to_png("./data/test_images/cluster_0003_B.fits", 
#             "./data/test_images/cluster_0003_B_64_ZScale.png",
#             conf.IMAGE_SIZE)
