import numpy as np
import shfl
import copy
import cv2
import csv
import os

from SDNET.model import SDNETModel
from shfl.private.federated_operation import FederatedData
from shfl.private.data import LabeledData

class FederatedSDNETModel(SDNETModel):          

    def transform_federated_data(self, federated_data):

        t_federated_data = copy.deepcopy(federated_data)

        for i in range(federated_data.num_nodes()):
            data = federated_data[i].query()._data
            labels = federated_data[i].query()._label
            t_data, t_labels = self.transform_data(data, labels)
            t_federated_data[i].query()._data = t_data
            t_federated_data[i].query()._label = t_labels

        return t_federated_data
    
    def set_aggregator(self, aggregator):
        self._aggregator = aggregator

    def run_rounds_CIT(self, rounds, federated_data, test_data, test_label):
        def cit_builder():
            return self._cit_model
        
        cit_federated_government = shfl.federated_government.FederatedGovernment(cit_builder, federated_data, self._aggregator)
        hist_cit = cit_federated_government.run_rounds(rounds, test_data, test_label)
        self._cit_model = copy.deepcopy(cit_federated_government.global_model)

        return hist_cit

    def run_rounds_Classifier(self, rounds, federated_data, test_data, test_label, transform = True):
        def classifier_builder():
            return self._classifier_model

        if transform:
            t_test_data, t_test_label = self._cit_model.transform_data(test_data, test_label, self._lb1, self._lb2)
            t_federated_data = self.transform_federated_data(federated_data)
        else:
            t_test_data, t_test_label = test_data, test_label
            t_federated_data = federated_data
        
        classifier_federated_government = shfl.federated_government.FederatedGovernment(classifier_builder, t_federated_data, self._aggregator)
        hist_classifier = classifier_federated_government.run_rounds(rounds, t_test_data, t_test_label)
        self._classifier_model = copy.deepcopy(classifier_federated_government.global_model)

        return hist_classifier

    def run_rounds(self, rounds, federated_data, test_data, test_label):
        
        hist_cit = self.run_rounds_CIT(rounds, federated_data, test_data, test_label)
        hist_classifier = self.run_rounds_Classifier(rounds, federated_data, test_data, test_label)        

        return hist_cit, hist_classifier

    def get_federated_data_csv(self, data_path, csv_path):
        width=256
        height=256

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
        train_label = self._lb1.transform(train_label)
        train_label = np.array(train_label)
        
        test_data = np.array(test_data)
        test_label = self._lb1.transform(test_label)
        test_label = np.array(test_label)
        
        for n in range(num_nodes):
            federated_train_label[n] = self._lb1.transform(federated_train_label[n])
            federated_train_label[n] = np.array(federated_train_label[n])
            federated_train_data[n] = np.array(federated_train_data[n])
        
        #federated_train_data = np.array(federated_train_data)
        #federated_train_label = np.array(federated_train_label)
        
        federated_data = FederatedData()
        for node in range(num_nodes):
            node_data = LabeledData(federated_train_data[node], federated_train_label[node])
            federated_data.add_data_node(node_data)
        
        return federated_data, train_data, train_label, test_data, test_label, train_files, test_files, num_nodes