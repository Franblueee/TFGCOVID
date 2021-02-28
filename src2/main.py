import os
import shfl
import torch
import copy
import numpy as np

from sklearn.preprocessing import LabelBinarizer

from shfl.private import UnprotectedAccess
from CIT.model import CITModel
from utils import get_federated_data_csv, get_data_csv
from ClassifierModel import ClassifierModel

args = {"data_path":"../data/COVIDGR1.0-cropped", 
        "csv_path": "../partitions/partition_iid_3nodes_1.csv",
        "output_path": "../weights",
        "input_path": "",
        "model_name":"transferlearning.model", 
        "label_bin": "lb.pickle", 
        "batch_size": 8,
        "federated_rounds": 1,
        "epochs_per_FL_round": 100,
        "num_nodes": 3,
        "size_averaging": 1,
        "random_rotation": 0,
        "random_shift": 0, 
        "random_zoom": 0,
        "horizontal_flip": False,        
        "finetune": True,
        "train_network": True
        }

#lambda_values = [float(10**(-n)) for n in range(1, 10)]
lambda_values = [0.00075]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def cit_builder():    
    return CITModel(['N', 'P'], classifier_name = "resnet18", lambda_values = lambda_values, batch_size=args["batch_size"], epochs=args["epochs_per_FL_round"], device=device)

def classifier_builder():
    return ClassifierModel(batch_size=args["batch_size"], epochs=args["epochs_per_FL_round"], finetune = args["finetune"])

def get_transformed_data(federated_data, cit_federated_government, test_data, test_label, lb1, lb2):
    t_federated_data = copy.deepcopy(federated_data)

    for i in range(federated_data.num_nodes()):
        data_node = federated_data[i]
        t_data_node = t_federated_data[i]
        data = data_node.query()._data
        labels = data_node.query()._label
        t_data, t_labels = cit_federated_government.global_model.transform_data(data, labels, lb1, lb2)
        t_data_node.query()._data = t_data
        t_data_node.query()._label = t_labels

    t_test_data, t_test_label = cit_federated_government.global_model.transform_data(test_data, test_label, lb1, lb2)

    return t_federated_data, t_test_data, t_test_label

a = ['N', 'P']
b = ['NTN', 'NTP', 'PTP', 'PTN']
lb1 = LabelBinarizer()
lb2 = LabelBinarizer()
lb1.fit(a)
lb2.fit(b)

def run_federated_experiment():


    print("[INFO] Fetching federated data...")
    federated_data, train_data, train_label, test_data, test_label, train_files, test_files = get_federated_data_csv(args["data_path"], args["csv_path"], lb1)
    federated_data.configure_data_access(UnprotectedAccess())
    print("[INFO] done")

    aggregator = shfl.federated_aggregator.FedAvgAggregator()
    cit_federated_government = shfl.federated_government.FederatedGovernment(cit_builder, federated_data, aggregator)
    cit_federated_government.run_rounds(args["federated_rounds"], test_data, test_label)

    t_federated_data, t_test_data, t_test_label = get_transformed_data(federated_data, cit_federated_government, test_data, test_label, lb1, lb2)

    aggregator = shfl.federated_aggregator.FedAvgAggregator()
    classifier_federated_government = shfl.federated_government.FederatedGovernment(classifier_builder, t_federated_data, aggregator)
    classifier_federated_government.run_rounds(args["federated_rounds"], t_test_data, t_test_label)

    dict_labels = { 'PTP' : np.argmax(lb2.transform(['PTP'])[0]) , 'PTN' : np.argmax(lb2.transform(['PTN'])[0]) , 
                    'NTP' : np.argmax(lb2.transform(['NTP'])[0]) , 'NTN' : np.argmax(lb2.transform(['NTN'])[0])
                } 
    G_dict = cit_federated_government.global_model._G_dict

    for key, _ in G_dict.items():
        G_dict[key].to("cpu")

    classifier_federated_government.global_model.get_classification_report(test_files, dict_labels, G_dict)

def run_centralized_experiment():

    data, label, train_data, train_label, test_data, test_label, train_files, test_files = get_data_csv(args["data_path"], args["csv_path"], lb1)

    cit_model = cit_builder()
    cit_model.train(train_data, train_label)

    t_train_data, t_train_label = cit_model.transform_data(train_data, train_label, lb1, lb2)
    t_test_data, t_test_label = cit_model.transform_data(test_data, test_label, lb1, lb2)

    classifier_model = classifier_builder()
    classifier_model.train(t_train_data, t_train_label)

    dict_labels = { 'PTP' : np.argmax(lb2.transform(['PTP'])[0]) , 'PTN' : np.argmax(lb2.transform(['PTN'])[0]) , 
                    'NTP' : np.argmax(lb2.transform(['NTP'])[0]) , 'NTN' : np.argmax(lb2.transform(['NTN'])[0])
                  } 
    G_dict = cit_model._G_dict

    for key, _ in G_dict.items():
        G_dict[key].to("cpu")

    classifier_model.get_classification_report(test_files, dict_labels, G_dict)

run_centralized_experiment()