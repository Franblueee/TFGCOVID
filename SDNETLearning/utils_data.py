import numpy as np
import csv
import os
from imutils import paths
from shutil import copyfile
import copy

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

        write_csv_file(federated_data, test_image_paths, csv_file_path)

def write_csv_file(federated_data, test_image_paths, csv_file_path):
    num_nodes = len(federated_data)
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

def calculate_label(row):
    gravedad = row['Gravedad RALE']
    obs = row['Clase observada radiólogo']
    pcr = row['Clase PCR (VP, VN, FP, FN)']

    if pcr == 'VP' or pcr=='FN':
        label = 'P'
    else:
        label = 'N'

    """
    if gravedad=='NORMAL':
        if pcr == 'VP' or pcr=='FN':
            label = 'P'
        else:
            label = 'N'
    else:
        label = 'P'
    """

    return label

def classify_images(metadata_file, data_path, classified_data_path):
    with open(metadata_file, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        count = 1
        for row in csv_reader:
            tipo = row['Nombre de la imagen (ID-Fecha-Tipo)'].split('-')[-1]
            if (tipo == 'PA'):
                name = row['ID (10 dígitos)'] + '-' + row['Fecha radiografía (DDMMAA)']
                image_path = data_path + os.sep + name + '.jpg'

                label = calculate_label(row)

                new_image_path = classified_data_path + os.sep + label + os.sep + name + '.jpg'

                if os.path.isfile(image_path):
                    copyfile(image_path, new_image_path) 

def get_hospital_dict(data_path, metadata_file):
    hospital_dict = {"San Cecilio" : [], "MOTRIL" : [], "ELCHE" : [], "BAZA" : [], "Loja" : [], "La Zubia" : [], "Gran Capitán" : []}
    with open(metadata_file, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        count = 1
        for row in csv_reader:
            tipo = row['Nombre de la imagen (ID-Fecha-Tipo)'].split('-')[-1]

            if (tipo == 'PA'):
                name = row['ID (10 dígitos)'] + '-' + row['Fecha radiografía (DDMMAA)']
                #name = row['Nombre imagen: anonimizar(ID + fecha + Tipo)']
                procedencia = row['Procedencia']
                if procedencia == "SAN CECILIO":
                    procedencia = "San Cecilio"
                
                label = calculate_label(row)
                image_path = data_path + os.sep + label + os.sep + name + '.jpg'

                if os.path.isfile(image_path):
                    hospital_dict[procedencia].append(image_path)

    return hospital_dict

def get_rale_dict(data_path, metadata_file):
    niveles_rale_dict = {"NEGATIVE": [], "NORMAL-PCR+": [], "LEVE": [], "MODERADO": [], "GRAVE" : []}
    keys = [k for k in range(9)] + ['N']
    puntos_rale_dict = {k : [] for k in keys}
    with open(metadata_file, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        count = 1
        for row in csv_reader:

            tipo = row['Nombre de la imagen (ID-Fecha-Tipo)'].split('-')[-1]

            if (tipo == 'PA'):
                
                label = calculate_label(row)
                name = row['ID (10 dígitos)'] + '-' + row['Fecha radiografía (DDMMAA)']
                #name = row['Nombre imagen: anonimizar(ID + fecha + Tipo)']
                puntos = int(row['Puntos RALE'])
                gravedad = row['Gravedad RALE']
                pcr = row['Clase PCR (VP, VN, FP, FN)']
                if gravedad=='NORMAL':
                    if label=='P':
                        gravedad = "NORMAL-PCR+"
                    else:
                        gravedad = "NEGATIVE"

                if puntos == 0 and label=='N':
                    puntos = 'N'
                
                image_path = data_path + os.sep + label + os.sep + name + '.jpg'

                if os.path.isfile(image_path):
                    niveles_rale_dict[gravedad].append(image_path)
                    puntos_rale_dict[puntos].append(image_path)
    return niveles_rale_dict, puntos_rale_dict

def get_demograf_dict(data_path, metadata_file):
    demograf_dict = { "Hombres<=50": [], "Mujeres<=50" : [], "Hombres(50,65]" : [], "Mujeres(50,65]" : [], "Hombres>65" : [], "Mujeres>65" : [] }
    with open(metadata_file, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        count = 1
        for row in csv_reader:
            name = row['ID (10 dígitos)'] + '-' + row['Fecha radiografía (DDMMAA)']
            #name = row['Nombre imagen: anonimizar(ID + fecha + Tipo)']
            sexo = row['Sexo (M, V) ']
            edad = int(row['Edad'])
            
            tipo = row['Nombre de la imagen (ID-Fecha-Tipo)'].split('-')[-1]

            if (tipo == 'PA'):
                """
                label = 'N'          
                image_path = data_path + os.sep + label + os.sep + name + '.jpg'
                if not os.path.isfile(image_path):
                    label = 'P'
                    image_path = data_path + os.sep + label + os.sep + name + '.jpg'
                """
                label = calculate_label(row)
                image_path = data_path + os.sep + label + os.sep + name + '.jpg'

                if os.path.isfile(image_path):
                    
                    if edad <= 50:
                        if sexo == 'V':
                            demograf_dict["Hombres<=50"].append(image_path)                        
                        else:
                            demograf_dict["Mujeres<=50"].append(image_path)                      
                    elif edad > 65:
                        if sexo == 'V':
                            demograf_dict["Hombres>65"].append(image_path)                        
                        else:
                            demograf_dict["Mujeres>65"].append(image_path)                       
                    else:
                        if sexo == 'V':
                            demograf_dict["Hombres(50,65]"].append(image_path)                        
                        else:
                            demograf_dict["Mujeres(50,65]"].append(image_path)

    return demograf_dict

def federate_data_hospital(hospital_dict):
    federated_data = []
    test_data = []
    train_prop = 0.8
    for k in ["San Cecilio", "MOTRIL", "ELCHE", "BAZA"]:
        image_paths = hospital_dict[k]
        image_paths = shuffle(image_paths)
        train_dim = int(train_prop * len(image_paths))
        train_image_paths = image_paths[0:train_dim]
        test_image_paths = image_paths[train_dim:]

        federated_data.append(train_image_paths)
        test_data.extend(test_image_paths)

    return federated_data, test_data

def federate_data_nivelesrale(niveles_rale_dict):

    neg_imgs = shuffle(copy.deepcopy(niveles_rale_dict["NEGATIVE"]))
    num_pos = len(niveles_rale_dict["NORMAL-PCR+"]) + len(niveles_rale_dict["LEVE"]) + len(niveles_rale_dict["MODERADO"]) + len(niveles_rale_dict["GRAVE"])
    num_neg = len(neg_imgs)
    neg_imgs_dict = {}
    for k in ["NORMAL-PCR+", "LEVE", "MODERADO", "GRAVE"]:
        num = int((len(niveles_rale_dict[k])/num_pos) * num_neg)
        a = []
        for i in range(num):
            a.append(neg_imgs[-1])
            neg_imgs.pop()
        neg_imgs_dict[k] = a

    keys = ["NORMAL-PCR+", "LEVE", "MODERADO", "GRAVE"]
    while neg_imgs:
        q = np.random.randint(1,4)
        neg_imgs_dict[keys[q]].append(neg_imgs[-1])
        neg_imgs.pop()
    
    federated_data = []
    test_data = []
    train_prop = 0.8
    for k in ["NORMAL-PCR+", "LEVE", "MODERADO", "GRAVE"]:
        pos_imgs = niveles_rale_dict[k]
        total = []
        total.extend(pos_imgs)
        total.extend(neg_imgs_dict[k])
        train_dim = int(train_prop * len(total))
        total = shuffle(total)
        train_image_paths = total[0:train_dim]
        test_image_paths = total[train_dim:]

        federated_data.append(train_image_paths)
        test_data.extend(test_image_paths)

    return federated_data, test_data

def federate_data_puntosrale(puntos_rale_dict):
    neg_imgs = shuffle(copy.deepcopy(puntos_rale_dict["N"]))
    num_pos = 0
    for k in range(9):
        num_pos = num_pos + len(puntos_rale_dict[k])
    num_neg = len(neg_imgs)
    neg_imgs_dict = {}
    for k in range(9):
        num = int((len(puntos_rale_dict[k])/num_pos) * num_neg)
        a = []
        for i in range(num):
            a.append(neg_imgs[-1])
            neg_imgs.pop()
        neg_imgs_dict[k] = a

    while neg_imgs:
        q = np.random.randint(0,9)
        neg_imgs_dict[q].append(neg_imgs[-1])
        neg_imgs.pop()
    
    federated_data = []
    test_data = []
    train_prop = 0.8
    for k in range(9):
        pos_imgs = puntos_rale_dict[k]
        total = []
        total.extend(pos_imgs)
        total.extend(neg_imgs_dict[k])
        train_dim = int(train_prop * len(total))
        total = shuffle(total)
        train_image_paths = total[0:train_dim]
        test_image_paths = total[train_dim:]

        federated_data.append(train_image_paths)
        test_data.extend(test_image_paths)

    return federated_data, test_data

def federate_data_demograf(demograf_dict):

    federated_data = []
    test_data = []
    train_prop = 0.8
    for k in demograf_dict.keys():
        image_paths = demograf_dict[k]
        image_paths = shuffle(image_paths)
        train_dim = int(train_prop * len(image_paths))
        train_image_paths = image_paths[0:train_dim]
        test_image_paths = image_paths[train_dim:]

        federated_data.append(train_image_paths)
        test_data.extend(test_image_paths)

    return federated_data, test_data