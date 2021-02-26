import os
import sys
import math
import random 
import shutil

def initFolders(dir_path, split_dir_path):

    if not os.path.exists(split_dir_path):
        os.makedirs(split_dir_path)

    for t in ['train', 'test']:
        if not os.path.exists(split_dir_path + os.sep + t):
            os.makedirs(split_dir_path + os.sep + t)
    
    for sub_dir in os.listdir(dir_path):
        for t in ['train', 'test']:
            if not os.path.exists(split_dir_path + os.sep + t + os.sep + sub_dir):
                os.makedirs(split_dir_path + os.sep + t + os.sep + sub_dir)


def splitTrainTest_2(img_dir, split_dir, train_prop, val_prop):
    dir_path = os.path.join(os.getcwd(), img_dir)
    split_dir_path = os.path.join(os.getcwd(), split_dir)

    if not os.path.exists(split_dir_path):
        os.makedirs(split_dir_path)

    for t in ['train', 'test']:
        if not os.path.exists(split_dir_path + os.sep + t):
            os.makedirs(split_dir_path + os.sep + t)
    
    for sub_dir in os.listdir(dir_path):
        for t in ['train', 'test']:
            if not os.path.exists(split_dir_path + os.sep + t + os.sep + sub_dir):
                os.makedirs(split_dir_path + os.sep + t + os.sep + sub_dir)

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
    
def splitTrainTest_1(img_dir, split_dir, train_prop):
    dir_path = os.path.join(os.getcwd(), img_dir)
    split_dir_path = os.path.join(os.getcwd(), split_dir)

    if not os.path.exists(split_dir_path):
        os.makedirs(split_dir_path)

    for t in ['train', 'test']:
        if not os.path.exists(split_dir_path + os.sep + t):
            os.makedirs(split_dir_path + os.sep + t)
    
    for sub_dir in os.listdir(dir_path):
        for t in ['train', 'test']:
            if not os.path.exists(split_dir_path + os.sep + t + os.sep + sub_dir):
                os.makedirs(split_dir_path + os.sep + t + os.sep + sub_dir)

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
        idx_permutation = random.sample(range(len(images)), len(images))

        if test_batch == 0:
            print(sub_dir + " has less than " + (1-train_prop) + " images")
            sys.exit(1)

        test_idx = idx_permutation[0:test_batch+1]
        train_idx = idx_permutation[test_batch+1:]

        [shutil.copy(path_sub_dir + os.sep + images[i], split_dir_path + os.sep + "test" + os.sep + sub_dir) for i in test_idx]

        [shutil.copy(path_sub_dir + os.sep + images[i], split_dir_path + os.sep + "train" + os.sep + sub_dir) for i in train_idx]

def splitTrainTestVal(img_dir, split_dir, train_prop, val_prop):
    dir_path = os.path.join(os.getcwd(), img_dir)
    split_dir_path = os.path.join(os.getcwd(), split_dir)

    if not os.path.exists(split_dir_path):
        os.makedirs(split_dir_path)

    for t in ['train', 'test']:
        if not os.path.exists(split_dir_path + os.sep + t):
            os.makedirs(split_dir_path + os.sep + t)
    
    for sub_dir in os.listdir(dir_path):
        for t in ['train', 'test']:
            if not os.path.exists(split_dir_path + os.sep + t + os.sep + sub_dir):
                os.makedirs(split_dir_path + os.sep + t + os.sep + sub_dir)

    train_data = {'P': [], 'N': []}
    test_data = {'P': [], 'N': []}
    val_data = {'P': [], 'N': []}
    
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

        # Dividimos las imágenes encontradas en train, val y test

        test_batch = math.floor( len(images)*(1-train_prop) )
        val_batch = math.floor( len(images)*train_prop*val_prop )
        idx_permutation = random.sample(range(len(images)), len(images))

        if test_batch == 0:
            print(sub_dir + " has less than " + (1-train_prop) + " images")
            sys.exit(1)

        print("Splitting images in " + sub_dir)

        test_idx = idx_permutation[0:test_batch+1]
        val_idx = idx_permutation[test_batch+1: test_batch+1+val_batch+1]
        train_idx = idx_permutation[test_batch+1+val_batch+1:]

        for i in train_idx:
            train_data[sub_dir].append(images[i])
            shutil.copy(path_sub_dir + os.sep + images[i], split_dir_path + os.sep + "train" + os.sep + sub_dir)
        
        for i in val_idx:
            val_data[sub_dir].append(images[i])
            shutil.copy(path_sub_dir + os.sep + images[i], split_dir_path + os.sep + "train" + os.sep + sub_dir)

        for i in test_idx:
            test_data[sub_dir].append(images[i])
            shutil.copy(path_sub_dir + os.sep + images[i], split_dir_path + os.sep + "test" + os.sep + sub_dir)
    
    return train_data, test_data, val_data

def splitTransformed(img_dir, save_split_dir, train_data, test_data, val_data):

    img_dir_path = os.path.join( os.getcwd(), img_dir )

    save_split_dir_path = os.path.join( os.getcwd(), save_split_dir )
    if not os.path.exists(save_split_dir_path):
        os.makedirs(save_split_dir_path)

    switcher = {
        'train' : train_data, 
        'test' : test_data,
        'val' : val_data
    }

    for d in ['train', 'test', 'val']: #train, test, val
    #for d in os.listdir(split_dir_path): #train, test, val
        if not os.path.exists( save_split_dir_path + os.sep + d ):
            os.makedirs( save_split_dir_path + os.sep + d )
        
        for c in ['NTN', 'NTP', 'PTN', 'PTP']:
            destiny = save_split_dir_path + os.sep + d + os.sep + c
            if not os.path.exists(destiny):
                os.makedirs(destiny)
        
        dic = switcher[d]
        for c in ['P', 'N']: # P, N
            for img in dic[c]:
                name = os.path.splitext(img)[0]
                for q in ['P', 'N']:
                    trans_img = name+"_"+c+"T18" + q + ".png" # le pongo png porque CIT las saca en png
                    trans_img_path = img_dir_path + os.sep + trans_img
                    destiny = save_split_dir_path + os.sep + d + os.sep + c + "T" + q
                    #print(trans_img_path)
                    #print(destiny)
                    shutil.copy(trans_img_path, destiny )

def splitTrainVal(img_dir, val_prop):
    dir_path = os.path.join(os.getcwd(), img_dir)
    train_path = dir_path + os.sep + "train"
    val_path = dir_path + os.sep + "val"

    if not os.path.exists(dir_path + os.sep + "val"):
        os.makedirs(dir_path + os.sep + "val")

    for sub_dir in ['NTN', 'NTP', 'PTN', 'PTP']:

        if not os.path.exists(dir_path + os.sep + "val" + os.sep + sub_dir):
            os.makedirs(dir_path + os.sep + "val" + os.sep + sub_dir)

        path_sub_dir = train_path + os.sep + sub_dir
        files = os.listdir(path_sub_dir)
        
        images = []
        for f in files:
            if (f.endswith('.jpg') or f.endswith('.JPG') or f.endswith('.jpeg') or f.endswith('.JPEG') or f.endswith('.png')):
                images.append(f)
        
        if len(images) == 0:
            print("No images found in " + sub_dir)
            sys.exit(1)

        print("Splitting images in " + sub_dir)

        # Dividimos las imágenes encontradas

        val_batch = math.floor( len(images)*val_prop )
        idx_permutation = random.sample(range(len(images)), len(images))

        print(val_batch)

        if val_batch == 0:
            print(sub_dir + " has less than " + str(len(images)*val_prop) + " images")
            sys.exit(1)

        val_idx = idx_permutation[0:val_batch]

        for x in val_idx:
            shutil.move(path_sub_dir + os.sep + images[x], val_path + os.sep + sub_dir)