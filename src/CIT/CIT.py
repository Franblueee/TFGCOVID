import argparse
import os
import shutil

import torch.utils.data
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

from CIT.data_utils import *
from CIT.model import Generator
from CIT.train_transform import train, transform_data

from PIL import Image

def CIT(data_dir, transform_dir, dataset, data_size, num_epochs, batch_size, classifier_name, lambda_value):
    data_dir_path = os.path.join( os.getcwd(), data_dir)
    data_train_dir_path = os.path.join( os.getcwd(), data_dir, "train" )
    transform_dir_path = os.path.join( os.getcwd(), transform_dir)

    if not os.path.exists(transform_dir_path):
        os.makedirs(transform_dir_path)

    train_set = ImageFolderWithPaths_noUps(data_train_dir_path, data_size,
                                           img_transforms=transforms.Compose([transforms.RandomHorizontalFlip(),
                                                                              transforms.RandomAffine(5),
                                                                              transforms.RandomRotation(5)]))
    
    G_dict = train(train_set, num_epochs, batch_size, lambda_value, classifier_name, dataset, data_size)
    transform_data(G_dict, data_size, data_dir_path, transform_dir_path, classifier_name)
