#!/home/ivan/anaconda3/bin/python3

import os
import sys

import numpy as np
#import pandas as pd

#import pydicom
import cv2
import matplotlib.pyplot as plt

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.losses import binary_crossentropy
from keras.utils import Sequence
from keras import backend as keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import PIL
import numpy as np
import matplotlib.pyplot as plt
import cv2

from glob import glob
#from tqdm import tqdm

SEGMENTATION_MODEL = "unet_lung_seg.hdf5"


def dice_coef(y_true, y_pred):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + 1)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def bounding_box(segmentationModel, image,threshold = 0):
    initial_width, initial_heith =image.size

    lamda1, lamda2 = initial_heith/512.0, initial_width/512.0
    initial_image = np.array(image)

    rgb_weights = [0.2989, 0.5870, 0.1140]
    image = np.array(image)
    rgb = False
    if image.shape[-1] == 3:
        image = np.dot(image[..., :3], rgb_weights)
        rgb =True

    image = cv2.resize(np.array(image), dsize=(512, 512) )
    image = image/255.0
    image = np.reshape(image, image.shape + (1,))
    image = np.reshape(image, (1,) + image.shape)
    prediction = segmentation_model.predict(image)
    prediction = prediction[0,:,:,0]

    heigth1, heigth2, width1, width2 = None,None,None, None
    i , j, k, l = 0,len(prediction)-1,0,len(prediction)-1
    maximoPredicho = np.max(prediction)
    if maximoPredicho != 1.0:
        maximoPredicho =maximoPredicho*(0.95)
    maximoPredicho = maximoPredicho*(1.0-threshold)
    while heigth1 == None and i < len(prediction):
        if np.any(prediction[:,i] >=maximoPredicho ):
            heigth1 = i
        i += 1
    while heigth2 == None and j >= heigth1:
        if np.any(prediction[:, j] >=maximoPredicho):
            heigth2 = j
        j -= 1
    
    while width1 == None and k < len(prediction):
        if np.any(prediction[k, :] >=maximoPredicho):
            width1 = k
        k += 1

    while width2 == None and l >= width1:
        if np.any(prediction[l, :] >=maximoPredicho):
            width2 = l
        l -= 1
    width1, width2, heigth1, heigth2 = int(width1*lamda1), int(width2*lamda1), int(heigth1*lamda2), int(heigth2*lamda2)

    pixelsH, pixelsW = int(0.025*(heigth2-heigth1)),int(0.025*(width2-width1))

    if rgb:
        finalImage = initial_image[max(width1-pixelsW,0):min(width2+pixelsW,len(initial_image)),max(0,heigth1-pixelsH):min(heigth2+pixelsH,len(initial_image[0])),:]
    else: 
        finalImage = initial_image[max(width1-pixelsW, 0):min(width2+pixelsW, len(initial_image)), max(0, heigth1-pixelsH):min(heigth2+pixelsH, len(initial_image[0]))]
    return finalImage

    


if __name__ == "__main__":
    
    databaseSource = None
    databaseSave = None
    threshold = 0.02
    
    """
    if len(sys.argv) < 2:
        databaseSource = "/home/ivan/Documentos/MEGA/COVID-19/SanCecilio/PA_PvsNvsD/"
    else:
        databaseSource = sys.argv[1]
    
    if len(sys.argv) < 3:
        databaseSave = "/home/ivan/Documentos/MEGA/COVID-19/SanCecilio/PA_PvsNvsDCropped/"
    else:
        databaseSave = sys.argv[2]
    """

    databaseSource = os.path.join(os.getcwd(), "images")
    databaseSave = os.path.join(os.getcwd(), "cropped")
    os.makedirs(databaseSave, exist_ok=True)
    #print(databaseSave)
    #print(databaseSource)

    segmentation_model = load_model(SEGMENTATION_MODEL,custom_objects={'dice_coef_loss': dice_coef_loss,'dice_coef': dice_coef})
    
    index = 0
    for directory in os.listdir(databaseSource):
        p = os.path.join(databaseSource, directory)
        #print(os.path.isdir(p))
        if os.path.isdir(p):
            img_save_dir = os.path.join(databaseSave, directory)
            os.makedirs(img_save_dir, exist_ok=True)
            listSource = os.listdir(p)
            for source in listSource:
                if source[-1]!= "~":
                    index+=1
                    #print(source,index)
                    img_path = os.path.join(p, source)
                    jpgImage = PIL.Image.open(img_path)
                    croppedImage = bounding_box(segmentation_model, jpgImage, threshold=threshold)
                    img_save_path = os.path.join(img_save_dir, source)
                    print("Cropped " + source)
                    plt.imsave(img_save_path, croppedImage, cmap="gray", vmin=0, vmax=255)
            