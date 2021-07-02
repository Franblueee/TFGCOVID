import numpy as np
import tensorflow as tf
import csv
import os
import cv2

from src3.SDNET.segment import LungsCropper
from src3.SDNET.classifier import ClassifierModel
from src3.SDNET.CIT.model import CITModel
from sklearn.preprocessing import LabelBinarizer

class SDNETmodel():

    def __init__(self, segmentation_path, lambda_value, epochs, batch_size, device):

        self._epochs = epochs
        self.batch_size = batch_size
        self._class_labels = ['N', 'P']
        self._transform_labels = ['NTN', 'NTP', 'PTP', 'PTN']

        """
        self._transform_labels = []
        for c in self._class_labels:
            for d in self._class_labels:
                self._transform_labels.append(c + "T" + d)
        """

        self._lb1 = LabelBinarizer()
        self._lb2 = LabelBinarizer()
        self._lb1.fit(self._class_labels)
        self._lb2.fit(self._transform_labels)

        dict_labels = {}
        for k in self._transform_labels:
            dict_labels[k] = np.argmax(self._lb2.transform([k])[0])
        for k in self._class_labels:
            dict_labels[k] = self._lb1.transform([k])[0][0]

        self._cropper = LungsCropper(segmentation_path)
        self._cit_model = CITModel(self._class_labels, "resnet18", lambda_value, batch_size, epochs, device)
        self._classifier_model = ClassifierModel(dict_labels, batch_size, epochs, finetune = True)

    def get_data_csv(self, data_path, csv_path):
        
        width=256
        height=256

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
        label = self._lb1.transform(label)

        train_data = np.array(train_data)
        train_label = np.array(train_label)
        train_label = self._lb1.transform(train_label)

        test_data = np.array(test_data)
        test_label = np.array(test_label)
        test_label = self._lb1.transform(test_label)
        
        return data, label, train_data, train_label, test_data, test_label, train_files, test_files


    def crop(self, data):
        return self._cropper.crop_data(data)

    def train_CIT(self, data, labels):
        self._cit_model.train(data, labels)

    def train_Classifier(self, data, labels, transform=True):
        if transform:
            t_data, t_labels = self._cit_model.transform_data(data, labels, self._lb1, self._lb2)
        else:
            t_data, t_labels = data, labels
        self._classifier_model.train(t_data, t_labels)

    def train(self, data, labels):
        self.train_CIT(data, labels)
        self.train_Classifier(data, labels)

    def predict_CIT(self, data):
        return self._cit_model.predict(data)

    def evaluate_CIT(self, data, labels):
        return self._cit_model.evaluate(data, labels)

    def predict(self, data, transform=True):
        #labels = [self._class_labels[0] for i in range(len(data))]
        #labels = self._lb1.transform(labels)
        if transform:
            t_data, _ = self._cit_model.transform_data(data, None, self._lb1, self._lb2)
        else:
            t_data, _ = data, None

        self._classifier_model.predict(data)

    def evaluate(self, data, labels, transform=True):
        if transform:
            t_data, t_labels = self._cit_model.transform_data(data, labels, self._lb1, self._lb2)
        else:
            t_data, t_labels = data, labels

        self._classifier_model.evaluate(t_data, t_labels)
