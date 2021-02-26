import tensorflow as tf
import shfl
import numpy as np
import os
import cv2

from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import classification_report

from CIT.data_utils import sample_loader


class ClassifierModel(shfl.model.DeepLearningModel):  

    def __init__(self, batch_size=1, epochs=1, finetune=True):
        
        resnet50 = tf.keras.applications.ResNet50(include_top = False, weights = 'imagenet', pooling = 'avg', input_tensor=Input(shape=(256, 256, 3)))
    
        if finetune:
            resnet50.trainable = False
        else: 
            resnet50.trainable = True
    
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
        
        self._batch_size = batch_size
        self._epochs = epochs

        self._model.compile(optimizer=self._optimizer, loss=self._criterion, metrics=self._metrics)  
    
    def train(self, data, labels):

        train_datagen = ImageDataGenerator(preprocessing_function = preprocess_input, validation_split=0.1)
        train_generator = train_datagen.flow(data, labels, batch_size=self._batch_size, subset='training')
        validation_generator = train_datagen.flow(data, labels, batch_size=self._batch_size, subset='validation')


        #early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='min')
        #early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=True)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_acc', patience = 10, restore_best_weights = True)
        self._model.fit(
            x=train_generator,
            steps_per_epoch= int(len(data)*0.9) // self._batch_size,
            validation_data = validation_generator,
            validation_steps = int(len(data)*0.1) // self._batch_size,
            epochs=self._epochs, 
            callbacks = [early_stopping]
        )
    
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
            
            x = sample_loader(image)
            tp = G_dict['P'](x)
            tp = tp[0].cpu().detach().numpy()
            tp = np.moveaxis(tp, 0, -1)
            
            tn = G_dict['N'](x)
            tn = tn[0].cpu().detach().numpy()
            tn = np.moveaxis(tn, 0, -1)

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
            
            
            # print('prediccion tp: ' + str(pred_tp))
            # print('prediccion tn: ' + str(pred_tn))

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

        tabla_preds[:,1] = true_labels
        tabla_preds[:,2] = preds
        #np.savetxt(save_preds_file, tabla_preds, fmt = '%1s', delimiter = ',')

        # Calculate accuracy
        acc_4 = sum(true_labels_4 == preds_4)/len(true_labels_4)
        print('Accuracy 4 clases: ' + str(acc_4))
        print('Numero de veces no concuerda: ' + str(no_concuerda))
        acc = sum(true_labels == preds)/len(true_labels)
        results = classification_report(true_labels, preds, digits = 5, output_dict = True)

        #if results['N']['recall'] >= 0.73 and results['P']['recall'] >= 0.73:
        #    self._model.save(save_model_file)

        print(results)
        print(acc)
