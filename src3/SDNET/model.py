from src3.segment import crop_data
import numpy as np
import tensorflow as tf
from src3.ClassifierModel import ClassifierModel
from src3.CIT.model import CITModel
from sklearn.preprocessing import LabelBinarizer

class SDNETmodel():

    def __init__(self, lambda_value, epochs, batch_size, device):

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

        self._cit_model = CITModel(self._class_labels, "resnet18", lambda_value, batch_size, epochs, device)
        self._classifier_model = ClassifierModel(dict_labels, batch_size, epochs, finetune = True)

    def crop(self, data):
        return crop_data(data)

    def train_CIT(self, data, labels):
        self._cit_model.train(data, labels)

    def train_SDNET(self, data, labels, transform=True):
        if transform:
            t_data, t_labels = self._cit_model.transform_data(data, labels, self._lb1, self._lb2)
        else:
            t_data, t_labels = data, labels
        self._classifier_model.train(t_data, t_labels)

    def train(self, data, labels):
        self.train_CIT(data, labels)
        self.train_SDNET(data, labels)

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
            t_data, _ = self._cit_model.transform_data(data, labels, self._lb1, self._lb2)
        else:
            t_data, _ = data, labels

        self._classifier_model.evaluate(data, labels)
