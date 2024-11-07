import argparse
import astropy.io.fits
import os

# Data location: https://www.canfar.net/storage/vault/list/PANDAS/PUBLIC/STACKS

# function to take a multi-extension and return the 'n th' extension.
def extract_extension(input_path, output_path, n):
    with astropy.io.fits.open(input_path, mode='readonly') as hdu:
        primary_header   = hdu[0].header
        extension_data   = hdu[n].data
        extension_header = hdu[n].header
        extension_header += primary_header
    astropy.io.fits.writeto(output_path, extension_data, extension_header,
                            output_verify='fix')

# extract a single image from a named MEF image  
# this one is 001
pointing = 'm001_g'
inpath = './data/PandAS/'+pointing+'.fit'
# Extracting 36 fit files out of one
for n in range(1,37):
    outpath = './data/PandAS/'+pointing+'_ccd'+str(n)+'.fit'
    extract_extension(inpath,outpath,n)