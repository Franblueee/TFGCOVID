import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="4"
import shfl
import torch 
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

from SDNET.model import SDNETmodel
from SDNET.CIT.model import CITModel

segmentation_path = "../weights/unet_lung_seg.hdf5"
lambda_value = 0.05
epochs = 1
batch_size = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def cit_builder():    
    return CITModel(['N', 'P'], "resnet18", 0.05, 2, 1, device)

data_path = "../data/COVIDGR1.0reducido-cropped" 
csv_path = "../partitions/partition_reducido.csv"

#data_path = "../data/Revisadas-Clasificadas-Recortadas" 
#csv_path = "../partitions/partition_iid_1nodes_1.csv"

def run():
    sdnet_model = SDNETmodel(lambda_value, epochs, batch_size, device, segmentation_path)

    data, label, train_data, train_label, test_data, test_label, train_files, test_files = sdnet_model.get_data_csv(data_path, csv_path)

    #cit_model = cit_builder()
    #cit_model.train(train_data, train_label)

    data = sdnet_model.crop(data)

    #sdnet_model.train_CIT(train_data, train_label)
    #sdnet_model.train_Classifier(train_data, train_label)

run()