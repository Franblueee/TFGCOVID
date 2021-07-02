
import os
import sys

import numpy as np

import cv2
import PIL

from tqdm import tqdm

from keras import backend as keras
from keras.models import load_model
from keras.preprocessing.image import load_img

SEGMENTATION_MODEL = "../weights" + os.sep + "unet_lung_seg.hdf5"

def dice_coef(y_true, y_pred):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + 1)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def cropImage(image, segmentation_model, threshold = 0):
    initial_heigth, initial_width, channels = image.shape

    lambda1, lambda2 = initial_heigth/512.0, initial_width/512.0
    initial_image = np.array(image)

    rgb_weights = [0.2989, 0.5870, 0.1140]
    rgb = False
    if image.shape[-1] == 3:
        image = np.dot(image[..., :3], rgb_weights)
        rgb = True

    image = cv2.resize(np.array(image), dsize=(512, 512) )
    image = image/255.0
    image = np.reshape(image, image.shape + (1,))
    image = np.reshape(image, (1,) + image.shape)

    pred_img = segmentation_model.predict(image)
    pred_img = pred_img[0,:,:,0]

    heigth1, heigth2, width1, width2 = None, None, None, None

    i , j, k, l = 0, len(pred_img)-1, 0, len(pred_img)-1
    maximoPredicho = np.max(pred_img)
    if maximoPredicho != 1.0:
        maximoPredicho = maximoPredicho*(0.95)
    maximoPredicho = maximoPredicho*(1.0-threshold)

    while heigth1 == None and i < len(pred_img):
        if np.any(pred_img[:,i] >= maximoPredicho ):
            heigth1 = i
        i += 1
    while heigth2 == None and j >= heigth1:
        if np.any(pred_img[:, j] >=maximoPredicho):
            heigth2 = j
        j -= 1
    
    while width1 == None and k < len(pred_img):
        if np.any(pred_img[k, :] >= maximoPredicho):
            width1 = k
        k += 1

    while width2 == None and l >= width1:
        if np.any(pred_img[l, :] >=maximoPredicho):
            width2 = l
        l -= 1
    width1, width2, heigth1, heigth2 = int(width1*lambda1), int(width2*lambda1), int(heigth1*lambda2), int(heigth2*lambda2)

    pixelsH, pixelsW = int(0.025*(heigth2-heigth1)),int(0.025*(width2-width1))

    if rgb:
        finalImage = initial_image[max(width1-pixelsW,0):min(width2+pixelsW,len(initial_image)),max(0,heigth1-pixelsH):min(heigth2+pixelsH,len(initial_image[0])),:]
    else: 
        finalImage = initial_image[max(width1-pixelsW, 0):min(width2+pixelsW, len(initial_image)), max(0, heigth1-pixelsH):min(heigth2+pixelsH, len(initial_image[0]))]
    return finalImage


def crop(image_dir, save_dir):

    threshold = 0.02

    image_dir_path = os.path.join( os.getcwd(), image_dir )
    save_dir_path = os.path.join( os.getcwd(), save_dir )

    if not os.path.exists(image_dir_path):
        print("Error: not found " + image_dir)
        sys.exit(1)

    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)

    segmentation_model = load_model(SEGMENTATION_MODEL, custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
    
    for c in os.listdir(image_dir_path):
        print("Cropping images from " + c)
        c_path = os.path.join(image_dir_path, c)
        save_c_path = os.path.join(save_dir_path, c)

        if not os.path.exists(save_c_path):
            os.makedirs(save_c_path)
        
        images = os.listdir(c_path)

        for img_name in tqdm(images):
            img_path = os.path.join(c_path, img_name)
            img = np.array(PIL.Image.open(img_path))
            cropped_img = cropImage(img, segmentation_model, threshold=threshold)
            img_save_path = os.path.join(save_c_path, img_name)
            cv2.imwrite(img_save_path, cropped_img)

def crop_data(data):
    new_data = []
    segmentation_model = load_model(SEGMENTATION_MODEL, custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
    threshold = 0.02
    for img in data:
        cropped_img = cropImage(img, segmentation_model, threshold=threshold)
        new_data.append(cropped_img)
    
    new_data = np.array(new_data)

    return new_data

image_dir = "../data/Revisadas-Clasificadas"
save_dir = "../data/Revisadas-Clasificadas-Recortadas"
crop(image_dir, save_dir)