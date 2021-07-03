import tensorflow as tf
import numpy as np
import shfl

from sklearn.metrics import classification_report

class ClassifierModel(shfl.model.DeepLearningModel):  

    def __init__(self, dict_labels, batch_size, epochs, finetune):
        
        #dict labels: letra -> bin
        self._dict_labels = dict_labels

        if finetune:
            resnet50 = tf.keras.applications.ResNet50(include_top = False, weights = 'imagenet', pooling = 'avg', input_tensor=tf.keras.layers.Input(shape=(224, 224, 3)))
        else: 
            resnet50 = tf.keras.applications.ResNet50(include_top = False, weights = None, pooling = 'avg', input_tensor=tf.keras.layers.Input(shape=(224, 224, 3)))
    
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

        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                                            preprocessing_function = tf.keras.applications.resnet50.preprocess_input,
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
        
        #probs = self._model.predict(data, batch_size=self._batch_size)

        preds = []
        preds_4 = []
        no_concuerda = 0
        for i in range(int(len(data)/2)):
            
            tn = data[2*i]
            tp = data[2*i+1]

            tn = np.expand_dims(tn, axis = 0)
            tp = np.expand_dims(tp, axis = 0)
            tp = tf.keras.applications.resnet50.preprocess_input(tp)
            tn = tf.keras.applications.resnet50.preprocess_input(tn)

            prob_tn = self._model.predict(tn)[0]
            prob_tp = self._model.predict(tp)[0]

            pred_tp = np.argmax(prob_tp)
            pred_tn = np.argmax(prob_tn)

            preds_4.append(pred_tn)
            preds_4.append(pred_tp)

            if pred_tp == self._dict_labels['NTP'] and pred_tn == self._dict_labels['NTN']:
                pred = self._dict_labels['N']
            elif pred_tp == self._dict_labels['PTP'] and pred_tn == self._dict_labels['PTN']:
                pred = self._dict_labels['P']
            else:
                no_concuerda = no_concuerda + 1
                # prob_p = prob_tp[0][dict['PTP']] + prob_tp[0][dict['PTN']] + prob_tn[0][dict['PTP']] + prob_tn[0][dict['PTN']]
                # prob_n = prob_tp[0][dict['NTP']] + prob_tp[0][dict['NTN']] + prob_tn[0][dict['NTP']] + prob_tn[0][dict['NTN']]
                prob_p = max(prob_tp[self._dict_labels['PTP']], prob_tp[self._dict_labels['PTN']], prob_tn[self._dict_labels['PTP']], prob_tn[self._dict_labels['PTN']])
                prob_n = max(prob_tp[self._dict_labels['NTP']], prob_tp[self._dict_labels['NTN']], prob_tn[self._dict_labels['NTP']], prob_tn[self._dict_labels['NTN']])
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

        labels_4 = np.array([np.argmax(l) for l in labels])
        labels_2 = []

        for i in range(int(len(labels)/2)):
            lab = np.argmax(labels[2*i])
            if lab == self._dict_labels['PTP'] or lab == self._dict_labels['PTN']:
                etiq = 'P'
            else:
                etiq = 'N'
            
            labels_2.append(self._dict_labels[etiq])
        
        labels_2 = np.array(labels_2)
        
        acc_4 = sum(labels_4 == preds_4)/len(labels_4)

        acc = sum(labels_2 == preds)/len(labels_2)

        cr = classification_report(labels_2, preds, digits = 5, output_dict = True)

        metrics = [acc, acc_4, no_concuerda, cr]

        return metrics
    
    def load(self, path):
        self._model.save(path)

    def save(self, path):
        self._model = tf.keras.models.load_model(path)
