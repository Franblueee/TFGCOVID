def get_percentage(federated_data):
    w = []
    total = 0

    for i in range(federated_data.num_nodes()):
        data = federated_data[i].query()._data
        w.append(len(data))
        total = total + len(data)
    
    for i in range(len(w)):
        w[i] = float(w[i]/total)

    return w

def imprimir_resultados(args, metrics_cit, metrics_sdnet, file, hist_CIT, hist_SDNET):
    f = open(file, "a")
    f.write("-------------------------------------------------------------------------------------\n")
    f.write("csv file: " + args["current_partition_file"] + "\n")
    f.write("batch_size: " + str(args["batch_size"])+ "\n")
    f.write("aggregator: " + str(args["aggregator"]) + "\n")
    f.write("CIT classifier: " + args["CIT"]["classifier_name"])
    f.write("CIT rounds: " + str(args["CIT"]["rounds"])+ "\n")
    f.write("SDNET rounds: " + str(args["Classifier"]["rounds"])+ "\n")
    f.write("CIT epochs: " + str(args["CIT"]["epochs"])+ "\n")
    f.write("SDNET epochs: " + str(args["Classifier"]["epochs"])+ "\n")
    f.write("num_nodes: " + str(args["num_nodes"])+ "\n")

    if metrics_cit:
        f.write("CIT Classifier Results:"+ "\n")
        f.write("Loss: {}".format(metrics_cit[0])+ "\n")
        f.write("Acc: {}".format(metrics_cit[1])+ "\n")
        cr = metrics_cit[2]
        f.write(str(cr['0']['precision']) + "\n")
        f.write(str(cr['0']['recall']) + "\n")
        f.write(str(cr['0']['f1-score']) + "\n")
        f.write(str(cr['1']['precision']) + "\n")
        f.write(str(cr['1']['recall']) + "\n")
        f.write(str(cr['1']['f1-score']) + "\n")
        f.write(str(metrics_cit[1]) + "\n")

    if metrics_sdnet:
        f.write("SDNET Classifier Results:"+ "\n")
        f.write("Acc: {}".format(metrics_sdnet[0])+ "\n")
        f.write("Acc_4: {}".format(metrics_sdnet[1])+ "\n")
        f.write("No concuerda: {}".format(metrics_sdnet[2])+ "\n")
        cr = metrics_sdnet[3]
        f.write(str(cr['0']['precision']) + "\n")
        f.write(str(cr['0']['recall']) + "\n")
        f.write(str(cr['0']['f1-score']) + "\n")
        f.write(str(cr['1']['precision']) + "\n")
        f.write(str(cr['1']['recall']) + "\n")
        f.write(str(cr['1']['f1-score']) + "\n")
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