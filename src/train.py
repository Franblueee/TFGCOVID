import shfl
import torch
import numpy as np
import tensorflow as tf
import torch
import json

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

from utils_train import  get_percentage, imprimir_resultados
from SDNETLearning.federated_model import FederatedSDNETModel
from shfl.private import UnprotectedAccess

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config_path = "./config.json"

with open(config_path) as config_buffer:
    args = json.loads(config_buffer.read())

sdnet = FederatedSDNETModel(args["CIT"]["lambda"], args["batch_size"], device, None)

for file in args["partition_files"]:

    args["current_partition_file"] = file

    sdnet.init_CIT(args["CIT"]["epochs"])
    sdnet.init_Classifier(args["Classifier"]["epochs"])

    federated_data, train_data, train_label, test_data, test_label, train_files, test_files, args["num_nodes"] = sdnet.get_federated_data_csv(args["data_path"], args["partition_path"] + file)
    federated_data.configure_data_access(UnprotectedAccess())

    if args['aggregator']=='fedavg':
        aggregator = shfl.federated_aggregator.FedAvgAggregator()
    else:
        percentage = get_percentage(federated_data)
        aggregator = shfl.federated_aggregator.WeightedFedAvgAggregator(percentage=percentage)

    sdnet.set_aggregator(aggregator)

    if args["CIT"]["load"]==1:
        sdnet._cit_model.load(args["load_weights_path"] + args["CIT"]["load_weights_name"])

    hist_cit = sdnet.run_rounds_CIT(args["CIT"]["rounds"], federated_data, test_data, test_label)

    if args["CIT"]["save"]==1:
        sdnet._cit_model.save(args["save_weights_path"] + args["CIT"]["save_weights_name"])

    if args["evaluate"]==1:
        metrics_cit = sdnet.evaluate_CIT(test_data, test_label)

    if args["Classifier"]["load"]==1:
        sdnet._classifier_model.load(args["load_weights_path"] + args["Classifier"]["load_weights_name"])

    hist_classifier = sdnet.run_rounds_Classifier(args["Classifier"]["rounds"], federated_data, test_data, test_label)

    if args["Classifier"]["save"]==1:
        sdnet._classifier_model.save(args["save_weights_path"] + args["CIT"]["save_weights_name"])

    if args["evaluate"]==1:
        metrics_sdnet = sdnet.evaluate(test_data, test_label)
        
    if args["results_path"] != "":
        imprimir_resultados(args, metrics_cit, metrics_sdnet, args["results_path"], hist_cit, hist_classifier)