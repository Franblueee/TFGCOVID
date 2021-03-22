import tensorflow as tf
import shfl
import numpy as np
import os
import cv2
import copy

from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import EarlyStopping

from torchvision.transforms import ToTensor, Resize

from sklearn.metrics import classification_report


class ClassifierModel(shfl.model.DeepLearningModel):  

    def __init__(self, G_dict, dict_labels, batch_size=1, epochs=1, finetune=True):
        
        self._G_dict = G_dict
        self._device = 'cuda'

        for class_name in ['P', 'N']:
            self._G_dict[class_name]= self._G_dict[class_name].to(self._device)

        #dict labels: letra -> bin
        #inv dict: bin -> letra
        self._dict_labels = dict_labels
    
        if finetune:
            resnet50 = tf.keras.applications.ResNet50(include_top = False, weights = 'imagenet', pooling = 'avg', input_tensor=Input(shape=(224, 224, 3)))
        else: 
            resnet50 = tf.keras.applications.ResNet50(include_top = False, weights = None, pooling = 'avg', input_tensor=Input(shape=(224, 224, 3)))
    
        # Add last layers
        x = resnet50.output
        x = tf.keras.layers.Dense(512, activation = 'relu')(x)
        predictions = tf.keras.layers.Dense(4, activation = 'softmax')(x)
    
        self._model = tf.keras.Model(inputs = resnet50.input, outputs = predictions)
        self._data_shape = self._model.layers[0].get_input_shape_at(0)[1:]
        self._labels_shape = self._model.layers[-1].get_output_shape_at(0)[1:]
    
        self._criterion = tf.keras.losses.CategoricalCrossentropy()
        self._optimizer = tf.keras.optimizers.SGD(lr = 1e-3, decay = 1e-6, momentum = 0.9, nesterov = True)
        self._metrics = [tf.keras.metrics.categorical_accuracy]
        #self._metrics = [tf.keras.metrics.Accuracy()]
        
        self._batch_size = batch_size
        self._epochs = epochs

        self._model.compile(optimizer=self._optimizer, loss=self._criterion, metrics=self._metrics)  
    
    def train(self, data, labels):

        train_datagen = ImageDataGenerator(
                                            preprocessing_function = preprocess_input,
                                            #rotation_range = 5,
                                            #width_shift_range = 0.5,
                                            #height_shift_range = 0.5,
                                            #horizontal_flip = True,
                                            validation_split=0.1
                                          )
        train_generator = train_datagen.flow(data, labels, batch_size=self._batch_size, subset='training', shuffle=True)
        validation_generator = train_datagen.flow(data, labels, batch_size=1, subset='validation', shuffle=False)


        #early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 10, verbose=1, restore_best_weights = True)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_categorical_accuracy', patience = 10, verbose=1, restore_best_weights = True)
        self._model.fit(
            x=train_generator,
            steps_per_epoch= int(len(data)*0.9) // self._batch_size,
            validation_data = validation_generator,
            validation_steps = int(len(data)*0.1),
            epochs=self._epochs, 
            callbacks = [early_stopping]
        )
    
    def predict(self, data):

        preds = []
        preds_4 = []
        no_concuerda = 0

        for image in data:

            if image.shape[0] != 256 or image.shape[1] != 256:
                image = cv2.resize(image, (256, 256))
            
            x = ToTensor()(image).float().unsqueeze(0).to(self._device)
            tp = self._G_dict['P'](x)
            tp = tp[0].cpu().detach().numpy()
            tp = np.moveaxis(tp, 0, -1)
            tp = cv2.resize(tp, dsize=(224, 224))

            tp= cv2.normalize(tp, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
            tp = cv2.resize(tp, dsize=(224, 224))
            tp = tp.astype(np.uint8)

            tn = self._G_dict['N'](x)
            tn = tn[0].cpu().detach().numpy()
            tn = np.moveaxis(tn, 0, -1)
            tn = cv2.resize(tn, dsize=(224, 224))

            tn = cv2.normalize(tn, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
            tn = cv2.resize(tn, dsize=(224, 224))
            tn = tn.astype(np.uint8)

            tp = np.expand_dims(tp, axis = 0)
            tn = np.expand_dims(tn, axis = 0)
            tp = preprocess_input(tp)
            tn = preprocess_input(tn)

            prob_tp = self._model.predict(tp)
            prob_tn = self._model.predict(tn)
            
            pred_tp = np.argmax(prob_tp)
            pred_tn = np.argmax(prob_tn)

            """
            max_tp = np.argmax(prob_tp)
            pred_tp = [0 for i in range(4)]
            pred_tp[max_tp] = 1
            pred_tp = np.array(pred_tp)

            max_tn = np.argmax(prob_tn)
            pred_tn = [0 for i in range(4)]
            pred_tn[max_tn] = 1
            pred_tn = np.array(pred_tn)
            """

            preds_4.append(pred_tp)
            preds_4.append(pred_tn)

            if pred_tp == self._dict_labels['NTP'] and pred_tn == self._dict_labels['NTN']:
                pred = self._dict_labels['N']
            elif pred_tp == self._dict_labels['PTP'] and pred_tn == self._dict_labels['PTN']:
                pred = self._dict_labels['P']
            else:
                no_concuerda = no_concuerda + 1
                # prob_p = prob_tp[0][dict['PTP']] + prob_tp[0][dict['PTN']] + prob_tn[0][dict['PTP']] + prob_tn[0][dict['PTN']]
                # prob_n = prob_tp[0][dict['NTP']] + prob_tp[0][dict['NTN']] + prob_tn[0][dict['NTP']] + prob_tn[0][dict['NTN']]
                prob_p = max(prob_tp[0][self._dict_labels['PTP']], prob_tp[0][self._dict_labels['PTN']], prob_tn[0][self._dict_labels['PTP']], prob_tn[0][self._dict_labels['PTN']])
                prob_n = max(prob_tp[0][self._dict_labels['NTP']], prob_tp[0][self._dict_labels['NTN']], prob_tn[0][self._dict_labels['NTP']], prob_tn[0][self._dict_labels['NTN']])
                if prob_p >= prob_n:
                    pred = self._dict_labels['P']
                else:
                    pred = self._dict_labels['N']

            preds.append(pred)

        preds = np.array(preds)
        preds_4 = np.array(preds_4)

        return preds, preds_4, no_concuerda

    def evaluate(self, data, labels):
        preds, preds_4, no_concuerda, = self.predict(data)

        new_labels = [ l[0] for l in labels ]

        new_labels = np.array(new_labels)

        labels_4 = []

        for i in range(len(labels)):
            if labels[i][0] == self._dict_labels['P']:
                etiq = 'P'
            else:
                etiq = 'N'
            
            etiq_tp = etiq + "TP"
            etiq_tn = etiq + "TN"
            labels_4.append(self._dict_labels[etiq_tp])
            labels_4.append(self._dict_labels[etiq_tn])

        labels_4 = np.array(labels_4)

        acc_4 = sum(labels_4 == preds_4)/len(labels_4)

        acc = sum(new_labels == preds)/len(labels)

        cr = classification_report(new_labels, preds, digits = 5, output_dict = True)

        metrics = [acc, acc_4, no_concuerda, cr]

        return metrics

    def get_classification_report(self, test_files, dict_labels, G_dict, save_model_file=None):
        true_labels = []
        preds = []
        no_concuerda = 0
        preds_4 = []
        true_labels_4 = []
        tabla_preds = np.empty((len(test_files), 3), dtype = '<U50')
        
        for i in range(len(test_files)):
            image_path = test_files[i]
            name = image_path.split(os.path.sep)[-1].split('.')[0]
            label = image_path.split(os.path.sep)[-2]
            
            true_labels.append(label)
            true_labels_4.append(dict_labels[label + "TP"])
            true_labels_4.append(dict_labels[label + "TN"])
            tabla_preds[i,0] = name
            
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (256, 256))
            
            x = ToTensor()(image).float().unsqueeze(0).to(self._device)
            tp = G_dict['P'](x)
            tp = tp[0].cpu().detach().numpy()
            tp = np.moveaxis(tp, 0, -1)
            tp = cv2.resize(tp, dsize=(224, 224))

            tn = G_dict['N'](x)
            tn = tn[0].cpu().detach().numpy()
            tn = np.moveaxis(tn, 0, -1)
            tn = cv2.resize(tn, dsize=(224, 224))

            tp = np.expand_dims(tp, axis = 0)
            tn = np.expand_dims(tn, axis = 0)
            tp = preprocess_input(tp)
            tn = preprocess_input(tn)

            prob_tp = self._model.predict(tp)
            prob_tn = self._model.predict(tn)
            
            pred_tp = np.argmax(prob_tp)
            pred_tn = np.argmax(prob_tn)
            preds_4.append(pred_tp)
            preds_4.append(pred_tn)
            
            #print(name)
            #print('prediccion tp: ' + str(pred_tp))
            #print('prediccion tn: ' + str(pred_tn))

            if pred_tp == dict_labels['NTP'] and pred_tn == dict_labels['NTN']:
                pred = 'N'
            elif pred_tp == dict_labels['PTP'] and pred_tn == dict_labels['PTN']:
                pred = 'P'
            else:
                no_concuerda = no_concuerda + 1
                # prob_p = prob_tp[0][dict['PTP']] + prob_tp[0][dict['PTN']] + prob_tn[0][dict['PTP']] + prob_tn[0][dict['PTN']]
                # prob_n = prob_tp[0][dict['NTP']] + prob_tp[0][dict['NTN']] + prob_tn[0][dict['NTP']] + prob_tn[0][dict['NTN']]
                prob_p = max(prob_tp[0][dict_labels['PTP']], prob_tp[0][dict_labels['PTN']], prob_tn[0][dict_labels['PTP']], prob_tn[0][dict_labels['PTN']])
                prob_n = max(prob_tp[0][dict_labels['NTP']], prob_tp[0][dict_labels['NTN']], prob_tn[0][dict_labels['NTP']], prob_tn[0][dict_labels['NTN']])
                if prob_p >= prob_n:
                    pred = 'P'
                else:
                    pred = 'N'

            preds.append(pred)

        true_labels = np.array(true_labels)
        preds = np.array(preds)
        true_labels_4 = np.array(true_labels_4)
        preds_4 = np.array(preds_4)

        print("preds")
        print(preds)

        print("preds_4")
        print(preds_4)

        tabla_preds[:,1] = true_labels
        tabla_preds[:,2] = preds
        #np.savetxt(save_preds_file, tabla_preds, fmt = '%1s', delimiter = ',')

        # Calculate accuracy
        acc_4 = sum(true_labels_4 == preds_4)/len(true_labels_4)
        print('Accuracy 4 clases: ' + str(acc_4))
        print('Numero de veces no concuerda: ' + str(no_concuerda))
        acc = sum(true_labels == preds)/len(true_labels)
        results = classification_report(true_labels, preds, digits = 5)

        #if results['N']['recall'] >= 0.73 and results['P']['recall'] >= 0.73:
        #    self._model.save(save_model_file)

        print(results)
        print(acc)
