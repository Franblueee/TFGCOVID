import json
import shfl
import numpy as np
import torch
import tensorflow as tf

from utils import get_federated_data_csv, get_transformed_data, get_percentage, imprimir_resultados
from ClassifierModel import ClassifierModel
from CIT.model import CITModel
from sklearn.preprocessing import LabelBinarizer

config_path = "./config.json"

with open(config_path) as config_buffer:
    args = json.loads(config_buffer.read())

lb1 = LabelBinarizer()
lb2 = LabelBinarizer()
lb1.fit(args["labels"])
lb2.fit(args["transform_labels"])

dict_labels = {}
for k in args["transform_labels"]:
    dict_labels[k] = np.argmax(lb2.transform([k])[0])
for k in args["labels"]:
    dict_labels[k] = lb1.transform([k])[0][0]

def cit_builder():    
    return CITModel(args["labels"], classifier_name = args["CIT"]["classifier_name"], lambda_value = args["lambda"], batch_size=args["batch_size"], epochs=args["CIT"]["epochs"], device=args["device"])

def classifier_builder():
    return ClassifierModel(dict_labels, batch_size=args["batch_size"], epochs=args["SDNET"]["epochs"], finetune = args["SDNET"]["finetune"])

hist_CIT = hist_SDNET = None
metrics_cit = metrics_sdnet = None

for file in args["partition_files"]:

    print("[INFO] Fetching federated data...")
    federated_data, train_data, train_label, test_data, test_label, train_files, test_files, num_nodes = get_federated_data_csv(args["data_path"], args["partition_path"] + file, lb1)
    federated_data.configure_data_access(shfl.private.UnprotectedAccess())
    print("[INFO] done")

    if args['aggregator']=='fedavg':
        aggregator = shfl.federated_aggregator.FedAvgAggregator()
    else:
        percentage = get_percentage(federated_data)
        aggregator = shfl.federated_aggregator.WeightedFedAvgAggregator(percentage=percentage)
    cit_federated_government = shfl.federated_government.FederatedGovernment(cit_builder, federated_data, aggregator)

    if args["CIT"]["load"]==1:
        cit_federated_government.global_model.load(args["load_weights_path"] + args["CIT"]["load_weights_name"])
    hist_CIT = cit_federated_government.run_rounds(args["CIT"]["rounds"], test_data, test_label)

    if args["CIT"]["save"]==1:
        cit_federated_government.global_model.save(args["save_weights_path"] + args["CIT"]["save_weights_name"])

    if args["evaluate"]==1:
        metrics_cit = cit_federated_government.global_model.evaluate(test_data, test_label)
        
    t_federated_data, t_test_data, t_test_label = get_transformed_data(federated_data, cit_federated_government, test_data, test_label, lb1, lb2)
    G_dict = cit_federated_government.global_model._G_dict
    classifier_federated_government = shfl.federated_government.FederatedGovernment(classifier_builder, t_federated_data, aggregator)
    
    if args["SDNET"]["load"]==1:
        classifier_federated_government.global_model.load(args["load_weights_path"] + args["SDNET"]["load_weights_name"])
    
    hist_SDNET = classifier_federated_government.run_rounds(args["rounds_SDNET"], t_test_data, t_test_label)

    if args["SDNET"]["save"]==1:
        classifier_federated_government.global_model.save(args["save_weights_path"] + args["SDNET"]["save_weights_name"])

    if args["evaluate"]==1:
        metrics_sdnet = classifier_federated_government.global_model.evaluate(t_test_data, t_test_label)
        
    if args["results_file"] != "":
        imprimir_resultados(args, metrics_cit, metrics_sdnet, args["results_path"] + args["results_file"], hist_CIT, hist_SDNET)