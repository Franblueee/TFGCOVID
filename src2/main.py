import os
import sys
import shfl
import torch
import numpy as np
import torch
import tensorflow as tf

from sklearn.preprocessing import LabelBinarizer

from shfl.private import UnprotectedAccess
from CIT.model import CITModel
from utils import get_federated_data_csv, get_data_csv, get_transformed_data, get_percentage
from ClassifierModel import ClassifierModel

from sklearn.model_selection import StratifiedKFold

args = {"data_path":"../data/Revisadas-Clasificadas-Recortadas", 
        "csv_dir": "../partitions/",
        "csv_path" : None,
        "weights_path": None,
        "results_file" : None,
        "train_CIT": 1, # 0: no train, 1: train from random weights, 2: train from loaded weights
        "batch_size": 8,
        "aggregator": 'wfedavg',
        "rounds_CIT":30,
        "rounds_SDNET":50, 
        "epochs_CIT": 10,
        "epochs_SDNET": 10,
        "num_nodes": None,
        "finetune": True,
        }

if len(sys.argv)>1:
    args["results_file"] = "../results/" + sys.argv[1]

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

def cit_builder(lambda_class = 0.05):    
    return CITModel(['N', 'P'], classifier_name = "resnet18", lambda_value = lambda_class, batch_size=args["batch_size"], epochs=args["epochs_CIT"], device=device)

def classifier_builder(G_dict):
    return ClassifierModel(G_dict, dict_labels, batch_size=args["batch_size"], epochs=args["epochs_SDNET"], finetune = args["finetune"])

def imprimir_configuracion():
    print("csv_path: " + args["csv_path"])
    print("batch_size: " + str(args["batch_size"]))
    print("aggregator: " + str(args["aggregator"]))
    print("rounds_CIT: " + str(args["rounds_CIT"]))
    print("rounds_SDNET: " + str(args["rounds_SDNET"]))
    print("epochs_CIT: " + str(args["epochs_CIT"]))
    print("epochs_SDNET: " + str(args["epochs_SDNET"]))
    print("num_nodes: " + str(args["num_nodes"]))
    print("finetune: " + str(args["finetune"]))

def imprimir_classification_report(f, cr):
    f.write(str(cr['0']['precision']) + "\n")
    f.write(str(cr['0']['recall']) + "\n")
    f.write(str(cr['0']['f1-score']) + "\n")
    f.write(str(cr['1']['precision']) + "\n")
    f.write(str(cr['1']['recall']) + "\n")
    f.write(str(cr['1']['f1-score']) + "\n")

def imprimir_resultados(metrics_cit, metrics_sdnet, file, hist_CIT=None, hist_SDNET=None):
    f = open(file, "a")
    f.write("-------------------------------------------------------------------------------------\n")
    f.write("csv_path: " + args["csv_path"] + "\n")
    f.write("batch_size: " + str(args["batch_size"])+ "\n")
    f.write("aggregator: " + str(args["aggregator"]) + "\n")
    f.write("rounds_CIT: " + str(args["rounds_CIT"])+ "\n")
    f.write("rounds_SDNET: " + str(args["rounds_SDNET"])+ "\n")
    f.write("epochs_CIT: " + str(args["epochs_CIT"])+ "\n")
    f.write("epochs_SDNET: " + str(args["epochs_SDNET"])+ "\n")
    f.write("num_nodes: " + str(args["num_nodes"])+ "\n")
    f.write("finetune: " + str(args["finetune"])+ "\n")

    if metrics_cit:
        f.write("CIT Classifier Results:"+ "\n")
        f.write("Loss: {}".format(metrics_cit[0])+ "\n")
        f.write("Acc: {}".format(metrics_cit[1])+ "\n")
        cr = metrics_cit[2]
        imprimir_classification_report(f, cr)
        f.write(str(metrics_cit[1]) + "\n")

    if metrics_sdnet:
        f.write("SDNET Classifier Results:"+ "\n")
        f.write("Acc: {}".format(metrics_sdnet[0])+ "\n")
        f.write("Acc_4: {}".format(metrics_sdnet[1])+ "\n")
        f.write("No concuerda: {}".format(metrics_sdnet[2])+ "\n")
        cr = metrics_sdnet[3]
        imprimir_classification_report(f, cr)
        f.write(str(metrics_sdnet[0]) + "\n")
        f.write(str(metrics_sdnet[1]) + "\n")

    if hist_CIT:
        f.write("CIT Metrics history:"+ "\n")
        for i in range(len(hist_CIT[0])-1):
            f.write("Metric " + str(i) + "\n")
            for n in range(len(hist_CIT)):
                f.write(str(hist_CIT[n][i]) + "\n")

    if hist_SDNET:
        f.write("SDNET Metrics history:"+ "\n")
        for i in range(len(hist_SDNET[0])-1):
            f.write("Metric " + str(i) + "\n")
            for n in range(len(hist_SDNET)):
                f.write(str(hist_SDNET[n][i]) + "\n")

    f.write("-------------------------------------------------------------------------------------\n")
    
    f.close()

def imprimir_hist(hist):

    for i in range(len(hist[0])-1):
        print("Metric " + str(i))
        for n in range(len(hist)):
            print(hist[n][i])

def run_federated_experiment():

    imprimir_configuracion()

    hist_CIT = hist_SDNET = None
    metrics_cit = metrics_sdnet = None

    print("[INFO] Fetching federated data...")
    federated_data, train_data, train_label, test_data, test_label, train_files, test_files, args["num_nodes"] = get_federated_data_csv(args["data_path"], args["csv_path"], lb1)
    federated_data.configure_data_access(UnprotectedAccess())
    print("[INFO] done")

    if args['aggregator']=='fedavg':
        aggregator = shfl.federated_aggregator.FedAvgAggregator()
    else:
        percentage = get_percentage(federated_data)
        aggregator = shfl.federated_aggregator.WeightedFedAvgAggregator(percentage=percentage)
    cit_federated_government = shfl.federated_government.FederatedGovernment(cit_builder, federated_data, aggregator)
    

    if args["train_CIT"]==0:
        if args["weights_path"]: 
            cit_federated_government.global_model.load(args["weights_path"])
    else:
        if args["train_CIT"]==2 and args["weights_path"]:
            cit_federated_government.global_model.load(args["weights_path"])
        
        hist_CIT = cit_federated_government.run_rounds(args["rounds_CIT"], test_data, test_label)
        imprimir_hist(hist_CIT)

        if args["weights_path"]:
            cit_federated_government.global_model.save(args["weights_path"])

    metrics_cit = cit_federated_government.global_model.evaluate(test_data, test_label)
    print("CIT Classifier Results:")
    print("Loss: {}".format(metrics_cit[0]))
    print("Acc: {}".format(metrics_cit[1]))
    print(metrics_cit[2])
    
    t_federated_data, t_test_data, t_test_label = get_transformed_data(federated_data, cit_federated_government, test_data, test_label, lb1, lb2)
    G_dict = cit_federated_government.global_model._G_dict
    classifier_federated_government = shfl.federated_government.FederatedGovernment(lambda : classifier_builder(G_dict), t_federated_data, aggregator)
    hist_SDNET = classifier_federated_government.run_rounds(args["rounds_SDNET"], t_test_data, t_test_label)

    metrics_sdnet = classifier_federated_government.global_model.evaluate(t_test_data, t_test_label)
    print("SDNET Classifier Results:")
    print("Acc: {}".format(metrics_sdnet[0]))
    print("Acc_4: {}".format(metrics_sdnet[1]))
    print("No concuerda: {}".format(metrics_sdnet[2]))
    print(metrics_sdnet[3])
    
    if args["results_file"] != None:
        imprimir_resultados(metrics_cit, metrics_sdnet, args["results_file"], hist_CIT, hist_SDNET)

def run_centralized_experiment():

    imprimir_configuracion()

    data, label, train_data, train_label, test_data, test_label, train_files, test_files = get_data_csv(args["data_path"], args["csv_path"], lb1)

    cit_model = cit_builder()
    cit_model.train(train_data, train_label)

    metrics_cit = cit_model.evaluate(test_data, test_label)
    print("CIT Classifier Results:")
    print("Loss: {}".format(metrics_cit[0]))
    print("Acc: {}".format(metrics_cit[1]))
    print(metrics_cit[2])

    #torch.cuda.empty_cache()

    t_train_data, t_train_label = cit_model.transform_data(train_data, train_label, lb1, lb2)
    #t_test_data, t_test_label = cit_model.transform_data(test_data, test_label, lb1, lb2)

    classifier_model = classifier_builder(cit_model._G_dict)
    classifier_model.train(t_train_data, t_train_label)

    metrics_sdnet = classifier_model.evaluate(test_data, test_label)
    print("SDNET Classifier Results:")
    print("Acc: {}".format(metrics_sdnet[0]))
    print("Acc_4: {}".format(metrics_sdnet[1]))
    print("No concuerda: {}".format(metrics_sdnet[2]))
    print(metrics_sdnet[3])

    
    outfile = "../results/centralized.txt"
    imprimir_resultados(metrics_cit, metrics_sdnet, outfile)

def run_cit():

    imprimir_configuracion()

    data, label, train_data, train_label, test_data, test_label, train_files, test_files = get_data_csv(args["data_path"], args["csv_path"], lb1)
        
    cit_model = cit_builder()
    cit_model.train(data, label)
    if args["weights_path"]:
        cit_model.save(args["weights_path"])
    metrics = cit_model.evaluate(test_data, test_label)
    print("CIT Classifier Results:")
    print("Loss: {}".format(metrics[0]))
    print("Acc: {}".format(metrics[1]))
    print(metrics[2])

def run_sdnet_crossval():

    print("SDNET LAMBDA CROSS VALIDATION")
    imprimir_configuracion()

    data, labels, train_data, train_label, test_data, test_label, train_files, test_files = get_data_csv(args["data_path"], args["csv_path"], lb1)
    
    args["folds"] = 1
    folds = 3
    #lambda_values = [0.05] + [float(10**(-n)) for n in range(1, 10)] 
    lambda_values = [float(10**(-n)) for n in range(3, 10)] 

    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1337)

    outfile = "../results/crossval.txt"

    for lambda_class in lambda_values:

        print("LAMBDA: " + str(lambda_class))


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

            cit_model = cit_builder(lambda_class)
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
            

        f = open(outfile, "a")
        f.write("CIT Classifier CV results for LAMBDA=" + str(lambda_class) + "\n")
        f.write("CV Loss: {}".format(np.mean(cit_cv_loss)) + "\n")
        f.write("CV Acc: {}".format(np.mean(cit_cv_acc)) + "\n")
        f.write("SDNET Classifier CV results for LAMBDA=" + str(lambda_class) + "\n")
        f.write("CV Acc: {}".format(np.mean(cv_acc)) + "\n")
        f.write("CV Acc_4: {}".format(np.mean(cv_acc_4)) + "\n")
        f.close()

"""
csv_dir = "../partitions/"
#csv_files = ["partition_iid_1nodes_1.csv", "partition_iid_1nodes_2.csv", "partition_iid_1nodes_3.csv", "partition_iid_1nodes_4.csv", "partition_iid_1nodes_5.csv"]
csv_files = ["partition_noniid_hospital_2.csv"]
for csv_file in csv_files:
    args["csv_path"] = csv_dir + csv_file
    print("-------------------------------------------------------------------------------------")
    print("FILE: " + csv_file)
    #run_centralized_experiment()
    run_cit()
    #run_sdnet_crossval()
    print("-------------------------------------------------------------------------------------")
"""

#csv_files = [ "partition_iid_"+str(n)+"nodes_"+str(id)+".csv" for n in [6] for id in [1,2,3]]

#csv_files = ["partition_noniid_hospital_1.csv", "partition_noniid_hospital_2.csv", "partition_noniid_hospital_3.csv"]
#csv_files = ["partition_noniid_nivelesrale_1.csv", "partition_noniid_nivelesrale_2.csv", "partition_noniid_nivelesrale_3.csv"]
#csv_files = ["partition_noniid_puntosrale_1.csv", "partition_noniid_puntosrale_2.csv", "partition_noniid_puntosrale_3.csv"]
#csv_files = ["partition_noniid_demograf_1.csv", "partition_noniid_demograf_2.csv", "partition_noniid_demograf_3.csv"]

csv_files = ["partition_iid_3nodes_1.csv", "partition_iid_6nodes_1.csv", "partition_iid_9nodes_1.csv"]


for csv_file in csv_files:
    print("hola")
    args["csv_path"] = args["csv_dir"] + csv_file
    print("-------------------------------------------------------------------------------------")
    print("FILE: " + csv_file)
    run_federated_experiment()
    print("-------------------------------------------------------------------------------------")
