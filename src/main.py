
from segment import *
from splitTrainTest import *
from CIT import CIT
from SDNET import *

import os
import random

if __name__ == "__main__":


    path = "data"+ os.sep

    #nombre = "COVIDGR1.0reducido"
    nombre = "COVIDGR1.0"

    image_dir = path + nombre + "-SinSegmentar"
    cropped_dir = path + nombre + "-Segmentadas"
    cropped_split_dir = path + nombre + "-SegmentadasSplit"
    transformed_dir = path + nombre + "Transformadas2"
    transformed_split_dir = path + nombre + "Transformadas2Split"

    SEED = 31416

    random.seed(SEED)

    train_prop = 0.8
    val_prop = 0.1

    #crop(image_dir, cropped_dir)
    
    train_data, test_data, val_data = splitTrainTestVal(cropped_dir, cropped_split_dir, train_prop, val_prop)
    
    data_size = 256 
    num_epochs = 100
    batch_size = 8
    classifier_name = 'resnet18'
    lambda_value = 0.0001

    CIT.CIT(cropped_split_dir, transformed_dir, nombre, data_size, num_epochs, batch_size, classifier_name, lambda_value)

    splitTransformed(transformed_dir, transformed_split_dir, train_data, test_data, val_data)
    
    img_rows = img_cols = 224
    batch_size = 8
    epochs = 100
    fine_tune = True
    random_shift = 0
    horizontal_flip = False
    random_zoom = 0
    random_rotation = 0
    save_model_file = "model.h5"
    use_weights = False
    reg_file = "tmp_weights.h5"
    save_preds_file = "save_preds.csv"

    transferLearning(transformed_split_dir, img_rows, img_cols, batch_size, epochs, fine_tune, random_shift, horizontal_flip, random_zoom, random_rotation, save_model_file, use_weights, reg_file, save_preds_file)
