import numpy as np
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from PIL import Image
import config as conf

def fits_to_png(fits_file, output_png, image_size):
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

fits_to_png("./data/test_images/cluster_0003_B.fits", 
            "./data/test_images/cluster_0003_B_64_ZScale.png",
            conf.IMAGE_SIZE)


# Read from a csv

# Function to read a csv, read up until the end magnitude for
# negatives
# Move to the start of the positives and read all of them

# For each one that is read, find the .fits file and convert it
# to the decided size