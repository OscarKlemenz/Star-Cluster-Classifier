from astropy.io import fits
import numpy as np
from PIL import Image

def fits_to_png(fitsFile, outputPng, targetSize=(64, 64)):
    """
    Convert a FITS file to PNG and resize it to a specific target size.
    
    Parameters:
    - fitsFile: path to the input FITS file.
    - outputPng: path to save the PNG file.
    - targetSize: tuple representing the desired image size, e.g., (64, 64).
    """

    hdul = fits.open(fitsFile)

    imageData = hdul[0].data
    hdul.close()

    imageData = np.log10(imageData + 1e-3)
    
    imageData -= np.min(imageData)  # Shift data to positive range
    imageData /= np.max(imageData)  # Normalize to 1
    imageData *= 255  # Scale to 255
    imageData = imageData.astype(np.uint8)

    img = Image.fromarray(imageData)

    img_resized = img.resize(targetSize)
    
    img_resized.save(outputPng)

    print(f"Converted {fitsFile} to {outputPng} with size {targetSize}.")

def fits_to_matrix(fitsFile, outputFile, format='npy'):
    """
    Convert a FITS file to a matrix and save it as a .npy or .csv file.
    
    Parameters:
    - fitsFile: path to the input FITS file.
    - outputFile: path to save the matrix file (without extension).
    - format: 'npy' for saving as a NumPy binary file, 'csv' for CSV.
    """
    
    hdul = fits.open(fitsFile)
    
    image_data = hdul[0].data
    hdul.close()

    # Save as a NumPy binary file
    if format == 'npy':
        np.save(outputFile, image_data)
        print(f"Matrix saved as {outputFile}.npy") 
    # Save as a CSV file
    elif format == 'csv':
        np.savetxt(f"{outputFile}.csv", image_data, delimiter=",")
        print(f"Matrix saved as {outputFile}.csv")
        
    else:
        print("Unsupported format. Please choose 'npy' or 'csv'.")

def png_to_matrix(pngFile, outputFile):
    """
    Convert a PNG image to a NumPy matrix (array).
    
    Parameters:
    - pngFile: Path to the PNG file.
    """
    image = Image.open(pngFile)

    imageMatrix = np.array(image)

    np.save(outputFile, imageMatrix)

png_to_matrix('./data/test_images/cluster_0003_B_64.png',
              './data/test_images/cluster_0003_B__64_matrix')

#fits_to_matrix('./data/test_images/cluster_0003_B.fits', 
#               './data/test_images/cluster_0003_B_matrix', 
#               format='npy')

# NEED TO THINK ABOUT LOG SCALE AND SIZE FOR THE MATRIX


# fits_to_png("./data/test_images/cluster_0003_B.fits", 
#            "./data/test_images/cluster_0003_B_256.png",
#            (256, 256))"
