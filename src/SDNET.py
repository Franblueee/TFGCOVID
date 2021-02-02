import tensorflow as tf
import numpy as np
import argparse
import os
import random
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from sklearn.metrics import classification_report

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def transferLearning(image_dir, imgs_rows, imgs_cols, batch_size, epochs,
                     fine_tune, random_shift, horizontal_flip,
                     random_zoom, random_rotation, save_model_file, use_weights,
                     reg_file, save_preds_file):

    image_dir_path = os.path.join( os.getcwd(), image_dir )
    
    train_path = image_dir_path + os.sep + "train"
    val_path = image_dir_path + os.sep + "val"
    test_path = image_dir_path + os.sep + "test"

    classes = os.listdir(image_dir_path + os.sep + "train")
    print(classes)

    # Create generators
    datagen_train = ImageDataGenerator(preprocessing_function = preprocess_input,
                                        rotation_range = random_rotation,
                                        width_shift_range = random_shift,
                                        height_shift_range = random_shift,
                                        zoom_range = random_zoom,
                                        horizontal_flip = horizontal_flip)

    datagen_val = ImageDataGenerator(preprocessing_function = preprocess_input)

    train_generator = datagen_train.flow_from_directory(
                        train_path,
                        target_size = (imgs_cols, imgs_rows),
                        batch_size = batch_size, 
                        classes = classes)

    val_generator = datagen_train.flow_from_directory(
                        val_path,
                        target_size = (imgs_cols, imgs_rows),
                        batch_size = batch_size,
                        classes = classes)

    test_files = []
    for c in classes:
        test_files.extend( os.listdir(test_path + os.sep + c ) ) 

    dict = train_generator.class_indices
    print(train_generator.class_indices)
    print(val_generator.class_indices)

    #train_labels = to_categorical(train_generator.classes, len(classes))
    #val_labels = to_categorical(val_generator.classes, len(classes))

    # Load model
    if fine_tune:
        resnet50 = tf.keras.applications.ResNet50(include_top = False,
                                                    weights = 'imagenet',
                                                    pooling = 'avg')
    else:
        resnet50 = tf.keras.applications.ResNet50(include_top = False,
                                                    weights = None,
                                                    pooling = 'avg')

    opt = tf.keras.optimizers.SGD(lr = 1e-3, decay = 1e-6, momentum = 0.9, nesterov = True)

    # Create dictionaries
    # dic = dict(zip(classes, range(len(classes))))
    # if use_weights:
    #     class_weight = {dic['COVID19']: 5.06, dic['NORMAL']: 1,
    #                     dic['PNEUMONIA']: 1}
    #     class_weight = {dic['COVID19']: 5.06, dic['NORMAL']: 1}
    #     class_weight = {dic['COVID19']: 5.06, dic['PNEUMONIA']: 1}
    # else:
    #     class_weight = {dic['COVID19']: 1, dic['NORMAL']: 1,
    #                     dic['PNEUMONIA']: 1}
    #     class_weight = {dic['COVID19']: 1, dic['NORMAL']: 1}
    #     class_weight = {dic['COVID19']: 1, dic['PNEUMONIA']: 1}

    # Creating EarlyStopping Callback
    earlyS = tf.keras.callbacks.EarlyStopping(monitor = 'val_accuracy', patience = 10, restore_best_weights = True)
    earlyS = tf.keras.callbacks.EarlyStopping(monitor = 'val_acc', patience = 10, restore_best_weights = True)


    # Add last layers
    x = resnet50.output
    # x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation = 'relu')(x)
    # x = tf.keras.layers.Dropout(0.5)(x)
    predictions = tf.keras.layers.Dense(len(classes), activation = 'softmax')(x)

    model = tf.keras.Model(inputs = resnet50.input, outputs = predictions)

    # Add regularization
    #model = add_regularization(model, regularizer = tf.keras.regularizers.l2(0.0005), weights_file = reg_file)

    # Compile and train the model
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_generator, steps_per_epoch = 864/batch_size, epochs = epochs, validation_data = val_generator, callbacks = [earlyS])

    # Get predicions for each pair of test images
    num_imgs_test = len(test_files) // 2
    true_labels = []
    preds = []
    no_concuerda = 0
    preds_4 = []
    true_labels_4 = []

    tabla_preds = np.empty((num_imgs_test, 3), dtype = '<U50')

    for i in range(num_imgs_test):
        current_img = test_files[0]
        name = str.split(current_img, '_')[0]
        class_transf_ext = str.split(current_img, '_')[1]
        class_transf = str.split(class_transf_ext, '.')[0]
        class_label = str.split(class_transf, 'T18')[0]
        transf = str.split(class_transf, 'T18')[1]
        
        tabla_preds[i,0] = name

        # tp --> transf positiva
        # tn --> transf negativa
        true_labels.append(class_label)
        tn = name + "_" + class_label + "T18" + "N.png"
        tp = name + "_" + class_label + "T18" + "P.png"
        
        test_files.remove(tn)
        test_files.remove(tp)
        
        true_labels_4.append(dict[class_label + "TP"])
        true_labels_4.append(dict[class_label + "TN"])

        tp = image.load_img(test_path + os.sep + class_label + "TP" + os.sep + tp, target_size = (imgs_cols, imgs_rows))
        tn = image.load_img(test_path + os.sep + class_label + "TN" + os.sep + tn, target_size = (imgs_cols, imgs_rows))
        tp = image.img_to_array(tp)
        tn = image.img_to_array(tn)
        tp = np.expand_dims(tp, axis = 0)
        tn = np.expand_dims(tn, axis = 0)
        tp = preprocess_input(tp)
        tn = preprocess_input(tn)

        prob_tp = model.predict(tp)
        prob_tn = model.predict(tn)

        pred_tp = np.argmax(prob_tp)
        pred_tn = np.argmax(prob_tn)
        preds_4.append(pred_tp)
        preds_4.append(pred_tn)
        # print('prediccion tp: ' + str(pred_tp))
        # print('prediccion tn: ' + str(pred_tn))

        if pred_tp == dict['NTP'] and pred_tn == dict['NTN']:
            pred = 'N'
        elif pred_tp == dict['PTP'] and pred_tn == dict['PTN']:
            pred = 'P'
        else:
            no_concuerda = no_concuerda + 1
            # prob_p = prob_tp[0][dict['PTP']] + prob_tp[0][dict['PTN']] + prob_tn[0][dict['PTP']] + prob_tn[0][dict['PTN']]
            # prob_n = prob_tp[0][dict['NTP']] + prob_tp[0][dict['NTN']] + prob_tn[0][dict['NTP']] + prob_tn[0][dict['NTN']]
            prob_p = max(prob_tp[0][dict['PTP']], prob_tp[0][dict['PTN']], prob_tn[0][dict['PTP']], prob_tn[0][dict['PTN']])
            prob_n = max(prob_tp[0][dict['NTP']], prob_tp[0][dict['NTN']], prob_tn[0][dict['NTP']], prob_tn[0][dict['NTN']])
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
    np.savetxt(save_preds_file, tabla_preds, fmt = '%1s', delimiter = ',')

    # Calculate accuracy
    acc_4 = sum(true_labels_4 == preds_4)/len(true_labels_4)
    print('Accuracy 4 clases: ' + str(acc_4))
    print('Numero de veces no concuerda: ' + str(no_concuerda))
    acc = sum(true_labels == preds)/len(true_labels)
    results = classification_report(true_labels, preds, digits = 5, output_dict = True)

    if results['N']['recall'] >= 0.73 and results['P']['recall'] >= 0.73:
        model.save(save_model_file)

    print(results)
    print(acc)
    return results, acc_4, no_concuerda

def crossValidation(base_dir, imgs_rows, imgs_cols, batch_size, epochs, fine_tune, random_shift, horizontal_flip, random_zoom, random_rotation, save_file, use_weights,
                    regularizer_file, save_preds, folds, results_file):

    recalls_N = []
    precisions_N = []
    f1s_N = []
    recalls_P = []
    precisions_P = []
    f1s_P = []
    accs_2 = []
    accs_4 = []
    no_concuerdas = []

    for fold in range(folds):
        reg_file = str.split(regularizer_file, '.h5')[0] + str(fold) + '.h5'
        save_preds_file = str.split(save_preds, '.csv')[0] + str(fold) + '.csv'
        save_model_file = str.split(save_file, '.h5')[0] + str(fold) + '.h5'
        print("Fold " + str(fold))
        partition_dir = base_dir + "partition" + str(fold)
        results, acc_4, no_concuerda = transferLearning(partition_dir, imgs_rows, imgs_cols, batch_size, epochs, fine_tune, random_shift, horizontal_flip, random_zoom, random_rotation, save_model_file, use_weights, reg_file, save_preds_file)
        with open(results_file, 'a') as f:
            f.write('\n')
            f.write('Fold ' + str(fold) + '\n')
            f.write('Recall N: ')
            f.write(str(results['N']['recall']))
            f.write('\n')
            f.write('Precision N: ')
            f.write(str(results['N']['precision']))
            f.write('\n')
            f.write('F1 N: ')
            f.write(str(results['N']['f1-score']))
            f.write('\n')
            f.write('Recall P: ')
            f.write(str(results['P']['recall']))
            f.write('\n')
            f.write('Precision P: ')
            f.write(str(results['P']['precision']))
            f.write('\n')
            f.write('F1 P: ')
            f.write(str(results['P']['f1-score']))
            f.write('\n')
            f.write('Acc 2: ')
            f.write(str(results['accuracy']))
            f.write('\n')
            f.write('Acc 4: ')
            f.write(str(acc_4))
            f.write('\n')
            f.write('No concuerda: ')
            f.write(str(no_concuerda))
            f.write('\n')

        recalls_N.append(results['N']['recall'])
        precisions_N.append(results['N']['precision'])
        f1s_N.append(results['N']['f1-score'])
        recalls_P.append(results['P']['recall'])
        precisions_P.append(results['P']['precision'])
        f1s_P.append(results['P']['f1-score'])
        accs_2.append(results['accuracy'])
        accs_4.append(acc_4)
        # no_concuerdas.append(no_concuerda)

        K.clear_session()

    # accs_2 = np.array(accs_2)
    # accs_4 = np.array(accs_4)
    # recalls_N = np.array(recalls_N)
    # precisions_N = np.array(precisions_N)
    # f1s_N = np.array(f1s_N)
    # recalls_P = np.array(recalls_P)
    # precisions_P = np.array(precisions_P)
    # f1s_P = np.array(f1s_P)

    return accs_2, accs_4, recalls_N, precisions_N, f1s_N, recalls_P, precisions_P, f1s_P