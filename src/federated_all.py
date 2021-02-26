# -*- coding: utf-8 -*-
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="7"
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import csv

import shfl

#"../data/COVIDGR1.0/centralized/cropped"

args = {"data_path":"../data/COVIDGR1.0/centralized/cropped", 
        "csv_path": "../data/partition1.csv",
        "output_path": "../weights",
        "input_path": "",
        "model_name":"transferlearning.model", 
        "label_bin": "lb.pickle", 
        "batch_size": 8,
        "federated_rounds": 2,
        "epochs_per_FL_round": 50,
        "num_nodes": 5,
        "size_averaging": 1,
        "random_rotation": 0,
        "random_shift": 0, 
        "random_zoom": 0,
        "horizontal_flip": False,        
        "finetune": True,
        "train_network": True}

a = ['N', 'P']
b = ['NTN', 'NTP', 'PTP', 'PTN']

lb1 = LabelBinarizer()
lb2 = LabelBinarizer()

lb1.fit(a)
lb2.fit(b)

from shfl.private.data import LabeledData
from shfl.private.federated_operation import FederatedData

def get_federated_data_csv(data_path, csv_path, width=256, height=256):
    
    num_nodes = 0
    
    with open(csv_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            current_node = int(row['node'])
            if current_node > num_nodes:
                num_nodes = current_node
    
    num_nodes = num_nodes + 1
        
    federated_train_data = [[] for i in range(num_nodes)]
    federated_train_label = [[] for i in range(num_nodes)]
    test_data = []
    test_label = []
    
    train_files = []
    test_files = []
    
    with open(csv_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            image_path = data_path + os.sep + row['class'] + os.sep + row['name'] + '.jpg'
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (width, height))
                
            if row['set'] == 'test':
                test_files.append(image_path)
                test_data.append(image)
                test_label.append(row['class'])
            else:
                train_files.append(image_path)
                node = int(row['node'])
                federated_train_data[node].append(image)
                federated_train_label[node].append(row['class'])
    
    test_data = np.array(test_data)
    test_label = np.array(test_label)
    test_label = lb1.transform(test_label)
    federated_train_data = np.array(federated_train_data)
    
    for n in range(num_nodes):
        federated_train_label[n] = lb1.fit_transform(federated_train_label[n])
        federated_train_data[n] = np.array(federated_train_data[n])
    
    federated_train_label = np.array(federated_train_label)
    
    federated_data = FederatedData()
    for node in range(num_nodes):
        node_data = LabeledData(federated_train_data[node], federated_train_label[node])
        federated_data.add_data_node(node_data)
    
    
    return federated_data, test_data, test_label, train_files, test_files

LABELS = ["N", "P"]
print("[INFO] training for labels: " + str(LABELS))

print("[INFO] Distributing the train set across the nodes...")
federated_data, test_data, test_label, train_files, test_files = get_federated_data_csv(args["data_path"], args["csv_path"])
print("[INFO] done")

"""from shfl.private import UnprotectedAccess

print(test_data.shape)
print(test_label.shape)
print(federated_data.num_nodes())
federated_data.configure_data_access(UnprotectedAccess())
a = federated_data[0].query()._data
print(test_label)

"""

from CITModel import CITModel
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def model_builder():    
    return CITModel(LABELS, classifier_name = "resnet18", lambda_value = 0.00075, batch_size=args["batch_size"], epochs=args["epochs_per_FL_round"], device=device)

aggregator = shfl.federated_aggregator.FedAvgAggregator()
federated_government = shfl.federated_government.FederatedGovernment(model_builder, federated_data, aggregator)
federated_government.run_rounds(args["federated_rounds"], test_data, test_label)

from shfl.private import UnprotectedAccess
from torchvision import transforms
from torchvision.transforms import ToPILImage
from torch.autograd import Variable
import matplotlib.pyplot as plt
from IPython.display import display
from PIL import Image
import cv2
from sklearn.preprocessing import LabelBinarizer

import copy


def sample_loader(sample):
    loader = transforms.Compose([transforms.ToTensor()])
    s = loader(sample).float()
    s = Variable(s, requires_grad=False)
    s = s.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return s.to(device) #assumes that you're using GPU

def show_img(G_dict, sample):
    x = sample_loader(sample)
    class_name = LABELS[0]
    y = G_dict[class_name](x)
    y = ToPILImage()(y[0].cpu().detach())
    #y.save("./prueba.png")
    display(y)
    
def transform_data(G_dict, class_names, data, labels):
    new_labels = []
    new_data = []
    for i in range(len(data)):
        sample = data[i]
        label = lb1.inverse_transform(labels[i])[0]
        x = sample_loader(sample)
        for class_name in class_names:
            y = G_dict[class_name](x)
            y = y[0].cpu().detach().numpy()
            y = np.moveaxis(y, 0, -1)
            new_data.append(y)
            new_label = str(label) + "T" + class_name
            new_labels.append(new_label)
    new_labels = lb2.transform(new_labels)
    
    return np.asarray(new_data), np.asarray(new_labels)


G_dict = federated_government.global_model._G_dict
for class_name in LABELS:
    G_dict[class_name]= G_dict[class_name].to(device)
federated_data.configure_data_access(UnprotectedAccess())

new_federated_data = copy.deepcopy(federated_data)

for i in range(federated_data.num_nodes()):
    data_node = federated_data[i]
    new_data_node = new_federated_data[i]
    data = data_node.query()._data
    labels = data_node.query()._label
    new_data, new_labels = transform_data(G_dict, LABELS, data, labels)
    new_data_node.query()._data = new_data
    new_data_node.query()._label = new_labels
    
    #print(data_node.query()._label)
    #print(new_data_node.query()._label)
    #data_node.query()._data = new_data
    #data_node.query()._label = new_labels

new_test_data, new_test_label = transform_data(G_dict, LABELS, test_data, test_label)

from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input

import tensorflow as tf


"""
datagen_train = ImageDataGenerator(
    preprocessing_function = preprocess_input,
    rotation_range = args["random_rotation"],
    width_shift_range = args["random_shift"],
    height_shift_range = args["random_shift"],
    zoom_range = args["random_zoom"],
    horizontal_flip = args["horizontal_flip"]
)
"""

train_datagen = ImageDataGenerator(preprocessing_function = preprocess_input, validation_split=0.1)



class TransferLearningModel(shfl.model.DeepLearningModel):    
    
    def train(self, data, labels):
        train_generator = train_datagen.flow(data, labels, batch_size=args["batch_size"], subset='training')

        validation_generator = train_datagen.flow(data, labels, batch_size=args["batch_size"], subset='validation')
        #self._check_data(data)
        #self._check_labels(labels)

        #early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='min')
        self._model.fit(
            x=train_generator,
            steps_per_epoch= int(len(data)*0.8) // args["batch_size"],
            validation_data = validation_generator,
            validation_steps = int(len(data)*0.2) // args["batch_size"],
            epochs=self._epochs
        )

def model_builder_2():
    
    resnet50 = tf.keras.applications.ResNet50(include_top = False, weights = 'imagenet', pooling = 'avg', input_tensor=Input(shape=(256, 256, 3)))
    
    if (args["finetune"]):
        resnet50.trainable = False
    else: 
        resnet50.trainable = True
    
    # Add last layers
    x = resnet50.output
    x = tf.keras.layers.Dense(512, activation = 'relu')(x)
    predictions = tf.keras.layers.Dense(4, activation = 'softmax')(x)
    
    model = tf.keras.Model(inputs = resnet50.input, outputs = predictions)
    
    criterion = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.SGD(lr = 1e-3, decay = 1e-6, momentum = 0.9, nesterov = True)
    metrics = [tf.keras.metrics.categorical_accuracy]
    
    return TransferLearningModel(model=model, criterion=criterion, optimizer=optimizer, metrics=metrics, epochs=args["epochs_per_FL_round"], batch_size = args["batch_size"])

from tensorflow.keras.models import load_model

# Train the network:
epochs_per_FL_round=args["epochs_per_FL_round"]
aggregator = shfl.federated_aggregator.FedAvgAggregator()
new_federated_government = shfl.federated_government.FederatedGovernment(model_builder_2, new_federated_data, aggregator)
new_federated_government.run_rounds(args["federated_rounds"], new_test_data, new_test_label)
#print("[INFO] saving model ...")
#federated_government.global_model._model.save( os.path.join(args["output_path"], args["model_name"]) )
print("[INFO] done")

from sklearn.metrics import classification_report

true_labels = []
preds = []
no_concuerda = 0
preds_4 = []
true_labels_4 = []
tabla_preds = np.empty((len(test_files), 3), dtype = '<U50')

model = new_federated_government.global_model._model

dict_labels = { 'PTP' : np.argmax(lb2.transform(['PTP'])[0]) , 'PTN' : np.argmax(lb2.transform(['PTN'])[0]) , 
                'NTP' : np.argmax(lb2.transform(['NTP'])[0]) , 'NTN' : np.argmax(lb2.transform(['NTN'])[0])
              } 

for i in range(len(test_files)):
    image_path = test_files[i]
    name = image_path.split(os.path.sep)[-1].split('.')[0]
    label = image_path.split(os.path.sep)[-2]
    
    true_labels.append(label)
    true_labels_4.append(dict_labels[label + "TP"])
    true_labels_4.append(dict_labels[label + "TN"])
    tabla_preds[i,0] = name
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256))
    
    x = sample_loader(image)
    tp = G_dict['P'](x)
    tp = tp[0].cpu().detach().numpy()
    tp = np.moveaxis(tp, 0, -1)
    
    tn = G_dict['N'](x)
    tn = tn[0].cpu().detach().numpy()
    tn = np.moveaxis(tn, 0, -1)

    tp = np.expand_dims(tp, axis = 0)
    tn = np.expand_dims(tn, axis = 0)
    tp = preprocess_input(tp)
    tn = preprocess_input(tn)

    prob_tp = model.predict(tp)
    prob_tn = model.predict(tn)
    
    pred_tp = np.argmax(prob_tp)
    pred_tn = np.argmax(prob_tn)
    preds_4.append(pred_tp)
    preds_4.append(pred_tn)
    
    
    # print('prediccion tp: ' + str(pred_tp))
    # print('prediccion tn: ' + str(pred_tn))

    if pred_tp == dict_labels['NTP'] and pred_tn == dict_labels['NTN']:
        pred = 'N'
    elif pred_tp == dict_labels['PTP'] and pred_tn == dict_labels['PTN']:
        pred = 'P'
    else:
        no_concuerda = no_concuerda + 1
        # prob_p = prob_tp[0][dict['PTP']] + prob_tp[0][dict['PTN']] + prob_tn[0][dict['PTP']] + prob_tn[0][dict['PTN']]
        # prob_n = prob_tp[0][dict['NTP']] + prob_tp[0][dict['NTN']] + prob_tn[0][dict['NTP']] + prob_tn[0][dict['NTN']]
        prob_p = max(prob_tp[0][dict_labels['PTP']], prob_tp[0][dict_labels['PTN']], prob_tn[0][dict_labels['PTP']], prob_tn[0][dict_labels['PTN']])
        prob_n = max(prob_tp[0][dict_labels['NTP']], prob_tp[0][dict_labels['NTN']], prob_tn[0][dict_labels['NTP']], prob_tn[0][dict_labels['NTN']])
        if prob_p >= prob_n:
            pred = 'P'
        else:
            pred = 'N'

    preds.append(pred)

true_labels = np.array(true_labels)
preds = np.array(preds)
true_labels_4 = np.array(true_labels_4)
preds_4 = np.array(preds_4)

print(preds)
print(true_labels)

tabla_preds[:,1] = true_labels
tabla_preds[:,2] = preds
#np.savetxt(save_preds_file, tabla_preds, fmt = '%1s', delimiter = ',')

# Calculate accuracy
acc_4 = sum(true_labels_4 == preds_4)/len(true_labels_4)
print('Accuracy 4 clases: ' + str(acc_4))
print('Numero de veces no concuerda: ' + str(no_concuerda))
acc = sum(true_labels == preds)/len(true_labels)
results = classification_report(true_labels, preds, digits = 5, output_dict = True)

#if results['N']['recall'] >= 0.73 and results['P']['recall'] >= 0.73:
#    model.save(save_model_file)

print(results)
print(acc)

