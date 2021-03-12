import numpy as np
import csv
import cv2
import os

from shfl.private.data import LabeledData
from shfl.private.federated_operation import FederatedData
from imutils import paths

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

    train_data = []
    train_label = []
    test_data = []
    test_label = []
    
    train_files = []
    test_files = []
    
    with open(csv_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            image_path = data_path + os.sep + row['class'] + os.sep + row['name'] + '.jpg'

            if not os.path.isfile(image_path):
                image_path = data_path + os.sep + row['class'] + os.sep + row['name'] + '.JPG'

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
                train_data.append(image)
                train_label.append(row['class'])
    
    train_data = np.array(train_data)
    train_label = np.array(train_label)
    train_label = label_binarizer.transform(train_label)

    test_data = np.array(test_data)
    test_label = np.array(test_label)
    test_label = label_binarizer.transform(test_label)
    
    
    for n in range(num_nodes):
        federated_train_label[n] = label_binarizer.transform(federated_train_label[n])
        federated_train_label[n] = np.array(federated_train_label[n])
        federated_train_data[n] = np.array(federated_train_data[n])
    
    #federated_train_data = np.array(federated_train_data)
    #federated_train_label = np.array(federated_train_label)
    
    federated_data = FederatedData()
    for node in range(num_nodes):
        node_data = LabeledData(federated_train_data[node], federated_train_label[node])
        federated_data.add_data_node(node_data)
    
    
    return federated_data, train_data, train_label, test_data, test_label, train_files, test_files, num_nodes

def get_data_csv(data_path, csv_path, label_binarizer, width=256, height=256):

    data = []
    label = []
    
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    
    train_files = []
    test_files = []
    
    with open(csv_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            image_path = data_path + os.sep + row['class'] + os.sep + row['name'] + '.jpg'

            if not os.path.isfile(image_path):
                image_path = data_path + os.sep + row['class'] + os.sep + row['name'] + '.JPG'
                
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (width, height))
                
            if row['set'] == 'test':
                test_files.append(image_path)
                test_data.append(image)
                test_label.append(row['class'])
            else:
                train_files.append(image_path)
                train_data.append(image)
                train_label.append(row['class'])
            data.append(image)
            label.append(row['class'])
    
    data = np.array(data)
    label = np.array(label)
    label = label_binarizer.transform(label)

    train_data = np.array(train_data)
    train_label = np.array(train_label)
    train_label = label_binarizer.transform(train_label)

    test_data = np.array(test_data)
    test_label = np.array(test_label)
    test_label = label_binarizer.transform(test_label)
    
    
    return data, label, train_data, train_label, test_data, test_label, train_files, test_files

def shuffle(list):
    randomize = np.arange(len(list))
    np.random.shuffle(randomize)
    new_list = [list[i] for i in randomize]

    return new_list

def federate_data_iid(data, num_nodes):
    data = shuffle(data)

    federated_data = [ [] for i in range(num_nodes) ]

    for i in range(len(data)):
        node = np.random.randint(num_nodes)
        federated_data[node].append(data[0])
        data.pop(0)
    
    return federated_data

def federate_data_iid_balanced(data, num_nodes):
    data = shuffle(data)

    size_per_node = len(data) // num_nodes
    rest = len(data) - num_nodes*size_per_node

    federated_data = []

    sum_used = 0

    for n in range(num_nodes):        
        x = [data[i] for i in range(sum_used, sum_used + size_per_node)]
        sum_used = sum_used + size_per_node
        federated_data.append(x)
    
    for n in range(rest):
        federated_data[n].append(data[sum_used])
        sum_used = sum_used+1
    
    return federated_data

def generate_iid_files(train_prop, num_nodes, seeds, path):
    for s in seeds:
        np.random.seed(s)

        csv_file_path = "../partitions/partition_iid_"+str(num_nodes)+"nodes_"+str(s)+".csv"

        image_paths = list(paths.list_images(path))

        train_dim = int(train_prop * len(image_paths))

        new_image_paths = shuffle(image_paths)

        train_image_paths = new_image_paths[0:train_dim]
        test_image_paths = new_image_paths[train_dim:]

        federated_data = federate_data_iid(train_image_paths, num_nodes)

        with open(csv_file_path, mode='w') as csv_file:
            #fieldnames = ['path', 'name', 'class', 'set', 'node']
            fieldnames = ['name', 'class', 'set', 'node']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            set_w = 'test'
            node = '-1'
            for image_path in test_image_paths:
                name = image_path.split(os.path.sep)[-1].split('.')[0]
                label = image_path.split(os.path.sep)[-2]
                #writer.writerow({'path': image_path, 'name': name, 'class': label, 'set': set_w, 'node': node})
                writer.writerow({'name': name, 'class': label, 'set': set_w, 'node': node})

            
            set_w = 'train'
            for n in range(num_nodes):
                node = str(n)
                for image_path in federated_data[n]:
                    name = image_path.split(os.path.sep)[-1].split('.')[0]
                    label = image_path.split(os.path.sep)[-2]
                    #writer.writerow({'path': image_path, 'name': name, 'class': label, 'set': set_w, 'node': node})
                    writer.writerow({'name': name, 'class': label, 'set': set_w, 'node': node})

def 