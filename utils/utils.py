import cv2
import numpy as np
import time
import torch
import matplotlib.pyplot as plt
import tifffile as tiff

import os 
import glob
from pathlib import Path

import plotly
from plotly import tools
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.offline as pyo
import plotly.io as pio
import plotly.graph_objects as go
import cv2
from tqdm.auto import tqdm

from paths_config import config_org as config

def elapsed_time(start_time):
    return time.time() - start_time

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def dice_score(A,B, smooth = 0):

    if (A==0).all() and (B==0).all(): return 1.0

    return 2*((A*B).sum())/(A.sum()+B.sum()+smooth)


def mask2rle(img): # encoder
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
 
def rle2mask(mask_rle, shape=(1600,256)): # decoder
    '''
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T


def plot_mask(image, mask, image_id):
    plt.figure(figsize=(10, 10))
    
    lab= cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(12,12))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl,a,b))

    # Converting image from LAB Color model to BGR color space
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    
    # subplot(nrows, ncols, index)
    plt.subplot(2, 2, 1)
    plt.imshow(image)
    plt.title(f"Image {image_id}", fontsize=14)

    plt.subplot(2, 2, 2)
    plt.imshow(enhanced_img)
    plt.title(f"Enhanced Image {image_id}", fontsize=14)
    
    plt.subplot(2, 2, 3)
    plt.imshow(mask, cmap="seismic", alpha =0.8)
    plt.title(f"Mask", fontsize=14)    
    
    plt.subplot(2, 2, 4)
    plt.imshow(enhanced_img)
    plt.imshow(mask, cmap="seismic", alpha=0.5)
    plt.title(f"Enhanced Image {image_id} + mask", fontsize=14)    


def read_image_mask(df, img_id):
    image = tiff.imread(str(config.TRAIN_IMAGES_PATH/ f"{img_id}.tiff"))
    
    mask = rle2mask(
        df[df["id"] == img_id]["rle"].values[0], 
        (image.shape[1], image.shape[0])
    )
    return image, mask

def image_info(df, img_id):
    organ = df[df['id'] == img_id]['organ'].values[0]
    pixel_size = df.loc[df['id'] == img_id]['pixel_size'].values[0]
    tissue_thickness = df.loc[df['id'] == img_id]['tissue_thickness'].values[0]
    age = df.loc[df['id'] == img_id]['age'].values[0]
    sex = df.loc[df['id'] == img_id]['sex'].values[0]
    image, mask_ = read_image_mask(df, img_id)
    print('\033[1m' + 'Case study : {}'.format(img_id) + '\033[0m')
    print('\n------------------------------\n')
    print('\033[1m' + 'General info: \n' + '\033[0m')
    print('\033[1m' + 'Organ: ' '\033[0m' + f'{organ}')   
    print('\033[1m' + 'Age: ' '\033[0m' + f'{age}')
    print('\033[1m' + 'Sex: ' '\033[0m' + f'{sex}')
    print('\n------------------------------\n')
    print('\033[1m' + 'Image + Mask info: \n' + '\033[0m')
    print('\033[1m' + 'Pixel size: ' '\033[0m' + f'{pixel_size}')
    print('\033[1m' + 'Tissue thickness: ' '\033[0m' + f'{tissue_thickness}')
    print('\033[1m' + 'Image shape: ' '\033[0m' + f'{image.shape}')
    print('\033[1m' + 'Mask shape: ' '\033[0m' + f'{mask_.shape}')
    plot_mask(image, mask_, img_id)
