import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import torch.nn.functional as F
import copy
import cv2


from shfl.model import TrainableModel
from torch import nn
from torchvision import models, transforms
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable

from CIT.loss import GeneratorLoss
from CIT.data_utils import sample_loader
from CIT.pytorchtools import EarlyStopping, CustomTensorDataset
from sklearn.model_selection import StratifiedKFold

from tqdm import tqdm

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report



class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)

        return block7

class CITModel(TrainableModel):
    def __init__(self, class_names, classifier_name = "resnet18", lambda_value = 0.05, batch_size=1, epochs=1, device="cpu"):
        
        self._class_names = class_names
        self._device = device
        self._lambda_value = lambda_value
        self._classifier_name = classifier_name
        self._batch_size = batch_size
        self._epochs = epochs

        self._class_weights = np.array([1.0, 1.0])

        #self._metrics = {'accuracy':accuracy_score, 'f1': f1_score, 'precision' : precision_score, 'recall' : recall_score}
        self._metrics = {'accuracy':accuracy_score, 'classification_report':classification_report}

        self._G_dict = self.create_generators()
        self._classifier = self.create_classifier()

    def create_generators(self):
        G_dict = {}
        
        for class_name in self._class_names:
            netG = Generator()
            G_dict[class_name] = netG.to(self._device)
        
        return G_dict

    def create_criterion_dict(self):
        G_criterion_dict = {}
        for class_name in self._class_names:
            generator_criterion = GeneratorLoss()
            G_criterion_dict[class_name] = generator_criterion.to(self._device)
        return G_criterion_dict

    def create_optimizers_dict(self, G_dict):
        optimizers_dict = {}
        
        for class_name in self._class_names:
            optimizers_dict[class_name] = optim.Adam(G_dict[class_name].parameters(), lr=0.0001, weight_decay=1e-4)
        
        return optimizers_dict
    
    def create_classifier(self):
        switcher = {
            'resnet50': models.resnet50,
            'resnet18': models.resnet18
        }
        # Get the function from switcher dictionary
        func = switcher[self._classifier_name]
        classifier = func(pretrained=True)

        # Freeze everything except fully connected
        ct = 0
        for child in classifier.children():
            #print("[INFO] FREEZING")
            ct += 1
            if ct < 7:
                 for param in child.parameters():
                    param.requires_grad = False

        # Change last layer to output N classes
        num_ftrs = classifier.fc.in_features
        classifier.fc = nn.Linear(num_ftrs, len(self._class_names))
        classifier.name = self._classifier_name
        classifier.to(self._device)

        return classifier

    def train(self, data, labels):
        
        unique, counts = np.unique(np.asarray(labels), return_counts=True)
        class_weights = np.max(counts)/counts
        class_weights /= np.max(class_weights)
        print("[INFO] weights = {}".format(class_weights))

        self._class_weights = class_weights

        my_transform = transforms.Compose( [ transforms.RandomHorizontalFlip(), transforms.RandomAffine(5), transforms.RandomRotation(5) ] )
        dataset = CustomTensorDataset(data, labels, 256, transform = my_transform)
        
        #my_transform = transforms.Compose( [ transforms.RandomHorizontalFlip(), transforms.RandomAffine(5), transforms.RandomRotation(5) ] )
        #dataset = CustomTensorDataset(torch.from_numpy(data), torch.from_numpy(labels), transform = my_transform)
        #train_dataset, val_dataset = random_split(dataset, [train_size, val_size] )

        best_G_dict = copy.deepcopy(self._G_dict)
        best_classifier = copy.deepcopy(self._classifier)

        train_size = int(0.9*len(data))
        val_size = len(data) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(dataset=train_dataset, batch_size=self._batch_size, num_workers=4)
        val_loader = DataLoader(dataset=val_dataset, batch_size=1, num_workers=4)

        valid_loss, y_true, y_pred, val_results = self.validate(val_loader, best_G_dict, best_classifier)
        best_acc = accuracy_score(y_true, y_pred)
        best_loss = valid_loss


        print("[INFO] Initial Valid Scores: ")
        print("Valid Acc = {}".format(best_acc))
        print("Valid Loss = {}".format(best_loss))

        print("[INFO] LAMBDA: {}".format(self._lambda_value))

        early_stopping = EarlyStopping(patience=5, verbose=True)
        early_stopping._best_score = - best_loss

        optimizer = optim.Adam(self._classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
        optimizers_dict = self.create_optimizers_dict(self._G_dict)
        criterion_classifier = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights).float().to(self._device))
        G_criterion_dict = self.create_criterion_dict()
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        for epoch in range(1, self._epochs+1):

            train_bar = tqdm(train_loader)
            running_results = {'batch_sizes': 0, 'd_loss': 0, 'd_corrects': 0, 'd_score': 0, 'g_score': 0}

            for class_name in self._class_names:
                running_results['g_loss_'+class_name] = 0
                    
            for class_name in self._class_names:
                self._G_dict[class_name].train()
            self._classifier.train()

            exp_lr_scheduler.step()

            for sample, target, label in train_bar:                                      
                        
                batch_size = sample.size(0)
                running_results['batch_sizes'] += batch_size

                input_img = Variable(target)
                input_img = input_img.to(self._device)
                label = label.to(self._device)
                        
                z = Variable(sample)
                z = z.to(self._device)
                        
                transformed_imgs = []
                unfold_labels = []
                # For each image in the batch, transform it N times (one per generator). Unfold GT label as many
                # times as transformed versions of an image (N times).
                for idx, ind_label in enumerate(label):
                    for class_name in self._class_names:
                        #print(z[idx].shape)
                        tr_image = self._G_dict[class_name](z[idx].unsqueeze(0))[0]
                        transformed_imgs.append(tr_image)
                        unfold_labels.append(ind_label.item())

                transformed_imgs = torch.stack(transformed_imgs)
                unfold_labels = torch.LongTensor(unfold_labels).to(self._device)

                # Predict transformed images
                classifier_outputs = self._classifier(transformed_imgs)
                classifier_outputs = classifier_outputs.to(self._device)
                loss_classifier = criterion_classifier(classifier_outputs, unfold_labels)

                # Optimize classifier
                optimizer.zero_grad()
                loss_classifier.backward(retain_graph=True)
                optimizer.step()


                # Optimize array of generators
                ce_toGenerator = {}  # Cross entropy loss for each generator
                for class_name in self._class_names:
                    current_label = self._class_names.index(class_name)
                    _, preds = torch.max(classifier_outputs, 1)
                    class_n_labels = unfold_labels[np.where(unfold_labels.cpu() == current_label)]
                    indexes = torch.from_numpy(np.where(unfold_labels.cpu() == current_label)[0]).to(self._device)
                    # Choose predictions for the current class
                    class_n_outputs = torch.index_select(classifier_outputs, 0, indexes)
                    if class_n_labels.shape[0] != 0:    # Maybe there are not samples of the current class in the batch
                        ce_toGenerator[self._class_names[current_label]] = criterion_classifier(class_n_outputs, class_n_labels)
                    else:
                        ce_toGenerator[class_name] = 0.0

                # Backprop one time per generator
                for idx, class_name in enumerate(self._class_names):
                    self._G_dict[class_name].zero_grad()
                    g_loss = G_criterion_dict[class_name](transformed_imgs[idx::len(self._class_names)], input_img, ce_toGenerator[class_name], float(self._lambda_value))
                    if idx < len(self._class_names)-1:
                        g_loss.backward(retain_graph=True)
                    else:
                        g_loss.backward()
                    optimizers_dict[class_name].step()

                # Re-do computations for obtaining loss after weight updates
                transformed_imgs = []
                for idx, ind_label in enumerate(label):
                    for class_name in self._class_names:
                        tr_image = self._G_dict[class_name](z[idx].unsqueeze(0))[0]
                        transformed_imgs.append(tr_image)
                transformed_imgs = torch.stack(transformed_imgs)
                for idx, class_name in enumerate(self._class_names):
                    g_loss = G_criterion_dict[class_name](transformed_imgs[idx::len(self._class_names)], input_img,
                                                            ce_toGenerator[class_name], float(self._lambda_value))
                    running_results['g_loss_'+class_name] += g_loss.item() * batch_size

                running_results['d_loss'] += loss_classifier.item() * batch_size
                _, preds = torch.max(classifier_outputs, 1)
                running_results['d_corrects'] += torch.sum(preds == unfold_labels.data).item()

                train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Acc_D: %.4f Loss_G_class1: %.4f Loss_G_class2: %.4f' % (
                    epoch, self._epochs, running_results['d_loss'] / (len(self._class_names)*running_results['batch_sizes']),
                    running_results['d_corrects'] / (len(self._class_names)*running_results['batch_sizes']),
                    running_results['g_loss_'+self._class_names[0]] / (len(self._class_names)*running_results['batch_sizes']),
                    running_results['g_loss_'+self._class_names[1]] / (len(self._class_names)*running_results['batch_sizes'])))

            valid_loss, y_true, y_pred, val_results = self.validate(val_loader, self._G_dict, self._classifier)
                
            curr_acc = accuracy_score(y_true, y_pred)
            print("\nValid Acc = {}".format(curr_acc))
            print("Valid Loss = {}".format(valid_loss))
            
            if curr_acc >= best_acc:
                best_acc = curr_acc
                best_loss = valid_loss
                best_G_dict = copy.deepcopy(self._G_dict)
                best_classifier = copy.deepcopy(self._classifier)
            
            """
            if valid_loss <= best_loss:
                best_acc = curr_acc
                best_loss = valid_loss
                best_G_dict = copy.deepcopy(G_dict)
                best_classifier = copy.deepcopy(classifier)
            """
            
            early_stopping(valid_loss)
            if early_stopping.early_stop:
                print("Early stopping, epoch " + str(epoch))
                break

        print("[INFO] Summary of training for LAMBDA = {} (best model values)".format(self._lambda_value))
        print("Valid Acc = {}".format(best_acc))
        print("Valid Loss = {}".format(best_loss))

        self._G_dict = copy.deepcopy(best_G_dict)
        self._classifier = copy.deepcopy(best_classifier)

    def validate(self, val_loader, G_dict, classifier, show=True):
        
        criterion_classifier = nn.CrossEntropyLoss(weight=torch.from_numpy(self._class_weights).float().to(self._device))

        valid_losses = []  # Store losses for each val img
        
        for class_name in self._class_names:
            G_dict[class_name].to(self._device)
        classifier.to(self._device)
            
        # Set models for predicting
        for class_name in self._class_names:
            G_dict[class_name].eval()
        classifier.eval()

        if show:
            val_bar = tqdm(val_loader)
            val_results = {'mse': 0, 'batch_sizes': 0, 'd_corrects': 0}
            y_true = []
            y_pred = []

            # Begin inference
            for sample, _, label in val_bar:
                #sample = sample.reshape( ( sample.shape[0], sample.shape[3], sample.shape[1], sample.shape[2] ) ).float()
                label = label.to(self._device)
                batch_size = sample.size(0)
                val_results['batch_sizes'] += batch_size
                sample.requires_grad = False
                sample = sample.to(self._device)

                classifier_outputs = []
                for idx, class_name in enumerate(self._class_names):
                    sr = G_dict[class_name](sample)
                    out = classifier(sr)
                    classifier_outputs.append(out)
                
                total_outputs = torch.cat(classifier_outputs, 1).squeeze()
                max_val, idx_max = torch.max(total_outputs, 0)
                preds = idx_max % len(self._class_names)
                
                labels_t = label
                if labels_t.shape[1] > 1:
                    labels_t = torch.argmax(labels_t, -1)
                    
                new_labels_t = torch.tensor((), dtype=torch.long).new_zeros( (2*labels_t.shape[0], labels_t.shape[1]) )
                for i in range(len(labels_t)):
                    new_labels_t[2*i] = labels_t[i]
                    new_labels_t[2*i+1] = labels_t[i]
                
                new_labels_t = new_labels_t.squeeze()
                
                a = torch.cat(classifier_outputs, 0).squeeze().to(self._device)
                b = new_labels_t.to(self._device)
                val_loss = criterion_classifier(a,b).item()
                
                valid_losses.append(val_loss)
                val_results['d_corrects'] += torch.sum(preds == label).item()
                y_true.append(label.item())
                y_pred.append(preds.item())

                val_bar.set_description(desc='[Validating]: Acc_D: %.4f' % (val_results['d_corrects'] / val_results['batch_sizes']))

            val_loss = np.average(np.asarray(valid_losses))

            return val_loss, y_true, y_pred, val_results

        else:
            y_true = []
            y_pred = []

            # Begin inference
            for sample, _, label in val_loader:
                #sample = sample.reshape( ( sample.shape[0], sample.shape[3], sample.shape[1], sample.shape[2] ) ).float()
                label = label.to(self._device)
                batch_size = sample.size(0)
                sample.requires_grad = False
                sample = sample.to(self._device)

                classifier_outputs = []
                for idx, class_name in enumerate(self._class_names):
                    sr = G_dict[class_name](sample)
                    out = classifier(sr)
                    classifier_outputs.append(out)
                
                total_outputs = torch.cat(classifier_outputs, 1).squeeze()
                max_val, idx_max = torch.max(total_outputs, 0)
                preds = idx_max % len(self._class_names)
                
                labels_t = label
                if labels_t.shape[1] > 1:
                    labels_t = torch.argmax(labels_t, -1)
                    
                new_labels_t = torch.tensor((), dtype=torch.long).new_zeros( (2*labels_t.shape[0], labels_t.shape[1]) )
                for i in range(len(labels_t)):
                    new_labels_t[2*i] = labels_t[i]
                    new_labels_t[2*i+1] = labels_t[i]
                
                new_labels_t = new_labels_t.squeeze()
                
                a = torch.cat(classifier_outputs, 0).squeeze().to(self._device)
                b = new_labels_t.to(self._device)
                val_loss = criterion_classifier(a,b).item()
                
                valid_losses.append(val_loss)
                y_true.append(label.item())
                y_pred.append(preds.item())

            val_loss = np.average(np.asarray(valid_losses))

            return val_loss, y_true, y_pred, None
        
    
    def predict(self, data):
        
        labels = np.array( [ 0 for i in range (len(data)) ] )

        dataset = CustomTensorDataset(data, labels, 256)
        data_loader = DataLoader(dataset=dataset, batch_size=1, num_workers=4)

        val_loss, y_true, y_pred, val_results = self.validate(data_loader, self._G_dict, self._classifier, show=False)
        
        return y_pred


    def evaluate(self, data, labels):
        
        dataset = CustomTensorDataset(data, labels, 256)
        data_loader = DataLoader(dataset=dataset, batch_size=1, num_workers=4)

        val_loss, y_true, y_pred, val_results = self.validate(data_loader, self._G_dict, self._classifier, show=False)
        
        
        acc = accuracy_score(y_pred, labels)
        cr = classification_report(y_pred, labels, digits=5, output_dict = True)

        metrics = [val_loss, acc, cr]
        
        return metrics

    def performance(self, data, labels):
        return self.evaluate(data, labels)[0]

    def get_model_params(self):

        with torch.no_grad():
            weights = []
            
            """
            class_weights = []
            for param in self._classifier.parameters():
                class_weights.append(param.cpu().data.numpy())
            weights.append(class_weights)
            """

            dict_classifier_tensor = self._classifier.state_dict()
            dict_classifier = { k : dict_classifier_tensor[k].cpu().numpy() for k in dict_classifier_tensor.keys() }
            
            weights.append(dict_classifier)

            for i in range(len(self._class_names)):
                class_name = self._class_names[i]
                """
                generator = self._G_dict[class_name] 
                gen_weights = []
                for param in generator.parameters():
                    gen_weights.append(param.cpu().data.numpy())
                weights.append(gen_weights)
                """
                dict_weights_tensor = self._G_dict[class_name].state_dict()
                dict_weights = { k : dict_weights_tensor[k].cpu().numpy() for k in dict_weights_tensor.keys() }
                weights.append(dict_weights)
            
            weights.append(self._class_weights)
        return weights

    def set_model_params(self, params):
        with torch.no_grad():
            
            """
            for ant, post in zip(self._classifier.parameters(), params[0]):
                #ant.data = torch.from_numpy(post).float()
                ant.copy_( torch.from_numpy(post).float() )
            """


            dict_classifier = params[0]
            dict_classifier_tensor = { k : torch.from_numpy(np.array(dict_classifier[k])) for k in dict_classifier.keys() }
            self._classifier.load_state_dict(dict_classifier_tensor)


            for i in range(len(self._class_weights)):
                class_name = self._class_names[i]
                dict_weights = params[i+1]
                dict_weights_tensor = { k : torch.from_numpy(np.array(dict_weights[k])) for k in dict_weights.keys() }
                self._G_dict[class_name].load_state_dict(dict_weights_tensor)
            
        self._class_weights = params[-1]

    def transform_data(self, data, labels, label_binarizer_1, label_binarizer_2):

        for class_name in self._class_names:
            self._G_dict[class_name]= self._G_dict[class_name].to(self._device)

        new_labels = []
        new_data = []
        for i in range(len(data)):
            sample = data[i]
            if sample.shape[0] != 256 or sample.shape[1] != 256:
                sample = cv2.resize(sample, (256, 256))
            label = label_binarizer_1.inverse_transform(labels[i])[0]
            x = transforms.ToTensor()(sample).float().unsqueeze(0).to(self._device)
            for i in range(len(self._class_names)):
                class_name = self._class_names[i]
                y = self._G_dict[class_name](x)
                y = y[0].cpu().detach().numpy()
                y = np.moveaxis(y, 0, -1)
                norm_y = cv2.normalize(y, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
                norm_y = cv2.resize(norm_y, dsize=(224, 224))
                norm_y = norm_y.astype(np.uint8)
                new_data.append(norm_y)
                new_label = str(label) + 'T' + class_name
                new_labels.append(new_label)
        
        new_labels = label_binarizer_2.transform(new_labels)
    
        return np.array(new_data), np.array(new_labels)
    
    def load(self, path):
        self._G_dict['N'].load_state_dict(torch.load(path+"CIT_G_N.pth"))
        self._G_dict['P'].load_state_dict(torch.load(path+"CIT_G_P.pth"))
        self._classifier.load_state_dict(torch.load(path+"CIT_C.pth"))

    def save(self, path):
        torch.save(self._G_dict['N'].state_dict(), path+"CIT_G_N.pth")
        torch.save(self._G_dict['P'].state_dict(), path+"CIT_G_P.pth")
        torch.save(self._classifier.state_dict(), path+"CIT_C.pth")
