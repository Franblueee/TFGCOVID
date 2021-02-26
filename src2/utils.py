import numpy as np
import csv
import cv2
import os

from shfl.private.data import LabeledData
from shfl.private.federated_operation import FederatedData

def get_federated_data_csv(data_path, csv_path, label_binarizer, width=256, height=256):
    
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
    test_label = label_binarizer.transform(test_label)
    federated_train_data = np.array(federated_train_data)
    
    for n in range(num_nodes):
        federated_train_label[n] = label_binarizer.transform(federated_train_label[n])
        federated_train_data[n] = np.array(federated_train_data[n])
    
    federated_train_label = np.array(federated_train_label)
    
    federated_data = FederatedData()
    for node in range(num_nodes):
        node_data = LabeledData(federated_train_data[node], federated_train_label[node])
        federated_data.add_data_node(node_data)
    
    
    return federated_data, test_data, test_label, train_files, test_files