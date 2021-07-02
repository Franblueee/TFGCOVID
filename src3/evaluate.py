import json
import shfl
import numpy as np
import sys

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

    print("[INFO] Fetching data...")
    federated_data, train_data, train_label, test_data, test_label, train_files, test_files, num_nodes = get_federated_data_csv(args["data_path"], args["partition_path"] + file, lb1)
    print("[INFO] done")

    cit_model = cit_builder()
    classifier_model = classifier_builder()

    if args["CIT"]["load_weights_file"]!="":
        cit_model.load(args["load_weights_path"] + args["CIT"]["load_weights_file"])
    else:
        sys.exit("Error: no has proporcionado archivo para cargar CIT")

    metrics_cit = cit_model.evaluate(test_data, test_label)

    if args["SDNET"]["load_weights_file"]!="":
        classifier_model.load(args["load_weights_path"])
    else:
        sys.exit("Error: no has proporcionado archivo para cargar CIT")
        
    t_test_data, t_test_label = cit_model.transform_data(test_data, test_label, lb1, lb2)
    metrics_sdnet = classifier_model.evaluate(t_test_data, t_test_label)
        
    if args["results_file"] != "":
        imprimir_resultados(args, metrics_cit, metrics_sdnet, args["results_file"], hist_CIT, hist_SDNET)