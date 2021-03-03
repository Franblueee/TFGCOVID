import os
import shfl
import torch
import copy
import numpy as np
import torch
import tensorflow as tf

from sklearn.preprocessing import LabelBinarizer

from shfl.private import UnprotectedAccess
from CIT.model import CITModel
from utils import get_federated_data_csv, get_data_csv
from ClassifierModel import ClassifierModel

from sklearn.model_selection import StratifiedKFold


args = {"data_path":"../data/COVIDGR1.0-Segmentadas", 
        "csv_path": None,
        "output_path": "../weights",
        "input_path": "",
        "batch_size": 8,
        "federated_rounds": 1,
        "epochs_per_FL_round": 50,
        "folds" : 1,
        "lambda_values" : [0.05],
        "num_nodes": 3,
        "finetune": True,
        }

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#lambda_values = [0.05]
#lambda_values = [1, 0.5, 0.1, 0.075, 0.05, 0.025, 0.01, 0.0075, 0.005, 0.0025, 0.001, 0.00075, 0.0005]
#lambda_values = [float(10**(-n)) for n in range(1, 10)] + [0.05]
#lambda_values = [float(10**(-5))]

a = ['N', 'P']
b = ['NTN', 'NTP', 'PTP', 'PTN']
lb1 = LabelBinarizer()
lb2 = LabelBinarizer()
lb1.fit(a)
lb2.fit(b)

dict_labels = { 'PTP' : np.argmax(lb2.transform(['PTP'])[0]) , 'PTN' : np.argmax(lb2.transform(['PTN'])[0]) , 
                'NTP' : np.argmax(lb2.transform(['NTP'])[0]) , 'NTN' : np.argmax(lb2.transform(['NTN'])[0]), 
                'P' : lb1.transform(['P'])[0][0], 'N' : lb1.transform(['N'])[0][0]
              } 

print(dict_labels)

def cit_builder():    
    return CITModel(['N', 'P'], classifier_name = "resnet18", lambda_values = args["lambda_values"], folds = args["folds"], batch_size=args["batch_size"], epochs=args["epochs_per_FL_round"], device=device)

def classifier_builder(G_dict):
    return ClassifierModel(G_dict, dict_labels, batch_size=args["batch_size"], epochs=args["epochs_per_FL_round"], finetune = args["finetune"])

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

    return t_federated_data

def imprimir_configuracion():
    print("csv_path: " + args["csv_path"])
    print("batch_size: " + str(args["batch_size"]))
    print("federated_rounds: " + str(args["federated_rounds"]))
    print("epochs_per_FL_round: " + str(args["epochs_per_FL_round"]))
    print("folds: " + str(args["folds"]))
    print("num_nodes: " + str(args["num_nodes"]))
    print("finetune: " + str(args["finetune"]))
    print("lambda: " + str(args["lambda_values"]))

def run_federated_experiment():

    imprimir_configuracion()

    print("[INFO] Fetching federated data...")
    federated_data, train_data, train_label, test_data, test_label, train_files, test_files = get_federated_data_csv(args["data_path"], args["csv_path"], lb1)
    federated_data.configure_data_access(UnprotectedAccess())
    print("[INFO] done")

    aggregator = shfl.federated_aggregator.FedAvgAggregator()
    cit_federated_government = shfl.federated_government.FederatedGovernment(cit_builder, federated_data, aggregator)
    cit_federated_government.run_rounds(args["federated_rounds"], test_data, test_label)

    metrics = cit_federated_government.global_model.evaluate(test_data, test_label)
    print("CIT Classifier Results:")
    print("Loss: {}".format(metrics[0]))
    print("Acc: {}".format(metrics[1]))
    print(metrics[2])

    t_federated_data = get_transformed_data(federated_data, cit_federated_government, test_data, test_label, lb1, lb2)


    aggregator = shfl.federated_aggregator.FedAvgAggregator()
    G_dict = cit_federated_government.global_model._G_dict

    classifier_federated_government = shfl.federated_government.FederatedGovernment(lambda : classifier_builder(G_dict), t_federated_data, aggregator)
    classifier_federated_government.run_rounds(args["federated_rounds"], test_data, test_label)

    metrics = classifier_federated_government.global_model.evaluate(test_data, test_label)
    print("SDNET Classifier Results:")
    print("Acc: {}".format(metrics[0]))
    print("Acc_4: {}".format(metrics[1]))
    print("No concuerda: {}".format(metrics[2]))
    print(metrics[3])


    """
    dict_labels = { 'PTP' : np.argmax(lb2.transform(['PTP'])[0]) , 'PTN' : np.argmax(lb2.transform(['PTN'])[0]) , 
                    'NTP' : np.argmax(lb2.transform(['NTP'])[0]) , 'NTN' : np.argmax(lb2.transform(['NTN'])[0])
                } 
    G_dict = cit_federated_government.global_model._G_dict

    for key, _ in G_dict.items():
        G_dict[key].to("cpu")

    classifier_federated_government.global_model.get_classification_report(test_files, dict_labels, G_dict)
    """

def run_centralized_experiment():

    imprimir_configuracion()

    data, label, train_data, train_label, test_data, test_label, train_files, test_files = get_data_csv(args["data_path"], args["csv_path"], lb1)

    cit_model = cit_builder()
    cit_model.train(train_data, train_label)

    metrics = cit_model.evaluate(test_data, test_label)
    print("CIT Classifier Results:")
    print("Loss: {}".format(metrics[0]))
    print("Acc: {}".format(metrics[1]))
    print(metrics[2])

    #torch.cuda.empty_cache()

    t_train_data, t_train_label = cit_model.transform_data(train_data, train_label, lb1, lb2)
    #t_test_data, t_test_label = cit_model.transform_data(test_data, test_label, lb1, lb2)

    classifier_model = classifier_builder(cit_model._G_dict)
    classifier_model.train(t_train_data, t_train_label)

    metrics = classifier_model.evaluate(test_data, test_label)
    print("SDNET Classifier Results:")
    print("Acc: {}".format(metrics[0]))
    print("Acc_4: {}".format(metrics[1]))
    print("No concuerda: {}".format(metrics[2]))
    print(metrics[3])

    """
    G_dict = cit_model._G_dict
    dict_labels = { 'PTP' : np.argmax(lb2.transform(['PTP'])[0]) , 'PTN' : np.argmax(lb2.transform(['PTN'])[0]) , 
                    'NTP' : np.argmax(lb2.transform(['NTP'])[0]) , 'NTN' : np.argmax(lb2.transform(['NTN'])[0]) 
                } 
    for key, _ in G_dict.items():
        G_dict[key].to("cpu")
    classifier_model.get_classification_report(test_files, dict_labels, G_dict)
    """

def run_cit():

    imprimir_configuracion()

    data, label, train_data, train_label, test_data, test_label, train_files, test_files = get_data_csv(args["data_path"], args["csv_path"], lb1)
        
    cit_model = cit_builder()
    cit_model.train(data, label)
    metrics = cit_model.evaluate(test_data, test_label)
    print("CIT Classifier Results:")
    print("Loss: {}".format(metrics[0]))
    print("Acc: {}".format(metrics[1]))
    print("F1: {}".format(metrics[2]))
    print("Precision: {}".format(metrics[3]))
    print("Recall: {}".format(metrics[4]))

def run_cit_crossval():

    imprimir_configuracion()

    args["lambda_values"] = [float(10**(-n)) for n in range(1, 10)] + [0.05]
    args["folds"] = 3

    data, label, train_data, train_label, test_data, test_label, train_files, test_files = get_data_csv(args["data_path"], args["csv_path"], lb1)
        
    cit_model = cit_builder()
    cit_model.train(data, label)
    metrics = cit_model.evaluate(test_data, test_label)
    print("CIT Classifier Results:")
    print("Loss: {}".format(metrics[0]))
    print("Acc: {}".format(metrics[1]))
    print("F1: {}".format(metrics[2]))
    print("Precision: {}".format(metrics[3]))
    print("Recall: {}".format(metrics[4]))

def run_sdnet_crossval():

    print("SDNET LAMBDA CROSS VALIDATION")
    imprimir_configuracion()

    data, labels, train_data, train_label, test_data, test_label, train_files, test_files = get_data_csv(args["data_path"], args["csv_path"], lb1)
    
    args["folds"] = 1
    folds = 3
    lambda_values = [float(10**(-n)) for n in range(1, 10)] + [0.05]

    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1337)

    for lambda_class in lambda_values:

        print("LAMBDA: " + str(lambda_class))

        args["lambda_values"] = [lambda_class]

        cit_cv_acc = []
        cit_cv_loss = []
        cv_acc = []
        cv_acc_4 = []

        for fold_idx, (train_index, test_index) in enumerate(kf.split(X=np.zeros(len(data)), y=labels)):
            
            print("FOLD: " + str(fold_idx))

            train_data = data[train_index]
            train_label = labels[train_index]
            test_data = data[test_index]
            test_label = labels[test_index]

            cit_model = cit_builder()
            cit_model.train(train_data, train_label)

            metrics = cit_model.evaluate(test_data, test_label)

            cit_cv_loss.append(metrics[0])
            cit_cv_acc.append(metrics[1])
            

            #torch.cuda.empty_cache()

            t_train_data, t_train_label = cit_model.transform_data(train_data, train_label, lb1, lb2)
            #t_test_data, t_test_label = cit_model.transform_data(test_data, test_label, lb1, lb2)

            classifier_model = classifier_builder(cit_model._G_dict)
            classifier_model.train(t_train_data, t_train_label)

            metrics = classifier_model.evaluate(test_data, test_label)

            cv_acc.append(metrics[0])
            cv_acc_4.append(metrics[1])
            

        print("CIT Classifier CV results for LAMBDA=" + str(lambda_class))
        print("CV Loss: {}".format(np.mean(cit_cv_loss)))
        print("CV Acc: {}".format(np.mean(cit_cv_acc)))
        
        print("SDNET Classifier CV results for LAMBDA=" + str(lambda_class))
        print("CV Acc: {}".format(np.mean(cv_acc)))
        print("CV Acc_4: {}".format(np.mean(cv_acc_4)))




csv_dir = "../partitions/"
#csv_files = [ ["partition_iid_1nodes_"+str(id)+".csv" for id in [1, 2, 3, 4, 5]] ] + [ [ "partition_iid_"+str(n)+"nodes_"+str(id)+".csv" for id in [1, 2, 3]] for n in [3, 4, 6] ]
#csv_files = ["partition_iid_1nodes_2.csv, partition_iid_1nodes_5csv"]
csv_files = [["partition_iid_1nodes_2.csv"]]
for csv_file in csv_files[0]:
    args["csv_path"] = csv_dir + csv_file
    print("-------------------------------------------------------------------------------------")
    print("FILE: " + csv_file)
    #run_centralized_experiment()
    #run_cit()
    run_sdnet_crossval()
    print("-------------------------------------------------------------------------------------")


"""
for n in [1, 2, 3]:
    for csv_file in csv_files[n]:
        args["csv_path"] = csv_dir + csv_file
        print("-------------------------------------------------------------------------------------")
        print("FILE: " + csv_file)
        run_federated_experiment()
        print("-------------------------------------------------------------------------------------")
"""