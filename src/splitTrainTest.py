import os
import sys
import math
import random 
import shutil


def initFolders(dir_path, split_dir_path):

    if not os.path.exists(split_dir_path):
        os.makedirs(split_dir_path)

    for t in ['train', 'test', 'val']:
        if not os.path.exists(split_dir_path + os.sep + t):
            os.makedirs(split_dir_path + os.sep + t)
    
    for sub_dir in os.listdir(dir_path):
        for t in ['train', 'test', 'val']:
            if not os.path.exists(split_dir_path + os.sep + t + os.sep + sub_dir):
                os.makedirs(split_dir_path + os.sep + t + os.sep + sub_dir)


def splitTrainTest(img_dir, split_dir, train_prop, val_prop):
    dir_path = os.path.join(os.getcwd(), img_dir)
    split_dir_path = os.path.join(os.getcwd(), split_dir)

    initFolders(dir_path, split_dir_path)
    
    for sub_dir in os.listdir(dir_path):
        path_sub_dir = dir_path + os.sep + sub_dir
        files = os.listdir(path_sub_dir)
        
        images = []
        for f in files:
            if (f.endswith('.jpg') or f.endswith('.JPG') or f.endswith('.jpeg') or f.endswith('.JPEG') or f.endswith('.png')):
                images.append(f)
        
        if len(images) == 0:
            print("No images found in " + sub_dir)
            sys.exit(1)

        print("Splitting images in " + sub_dir)

        # Dividimos las imágenes encontradas entre las carpetas de train, val y test

        test_batch = math.floor( len(images)*(1-train_prop) )
        val_batch = math.floor( len(images)*train_prop*val_prop )
        idx_permutation = random.sample(range(len(images)), len(images))

        if test_batch == 0:
            print(sub_dir + " has less than " + (1-train_prop) + " images")
            sys.exit(1)

        test_idx = idx_permutation[0:test_batch+1]
        val_idx = idx_permutation[test_batch+1: test_batch+1+val_batch+1]
        train_idx = idx_permutation[test_batch+1+val_batch+1:]

        [shutil.copy(path_sub_dir + os.sep + images[i], split_dir_path + os.sep + "test" + os.sep + sub_dir) for i in test_idx]

        [shutil.copy(path_sub_dir + os.sep + images[i], split_dir_path + os.sep + "val" + os.sep + sub_dir) for i in val_idx]

        [shutil.copy(path_sub_dir + os.sep + images[i], split_dir_path + os.sep + "train" + os.sep + sub_dir) for i in train_idx]
    

def splitTransformed(img_dir, save_split_dir, split_dir):

    img_dir_path = os.path.join( os.getcwd(), img_dir )
    split_dir_path = os.path.join( os.getcwd(), split_dir )

    save_split_dir_path = os.path.join( os.getcwd(), save_split_dir )
    if not os.path.exists(save_split_dir_path):
        os.makedirs(save_split_dir_path)

    for d in ['train', 'test', 'val']: #train, test, val
    #for d in os.listdir(split_dir_path): #train, test, val
        if not os.path.exists( save_split_dir_path + os.sep + d ):
            os.makedirs( save_split_dir_path + os.sep + d )
        
        for c in ['NTN', 'NTP', 'PTN', 'PTP']:
            destiny = save_split_dir_path + os.sep + d + os.sep + c
            if not os.path.exists(destiny):
                os.makedirs(destiny)
        
        for c in os.listdir( split_dir_path + os.sep + d ): # P, N
            for img in os.listdir( split_dir_path + os.sep + d + os.sep + c):
                name = os.path.splitext(img)[0]
                for q in ['P', 'N']:
                    trans_img = name+"_"+c+"T18" + q + ".png" # le pongo png porque CIT las saca en png
                    trans_img_path = img_dir_path + os.sep + trans_img
                    destiny = save_split_dir_path + os.sep + d + os.sep + c + "T" + q
                    #print(trans_img_path)
                    #print(destiny)
                    shutil.copy(trans_img_path, destiny )