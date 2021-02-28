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

from sklearn.metrics import accuracy_score


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
    def __init__(self, class_names, classifier_name = "resnet18", lambda_values = [0.00075], batch_size=1, epochs=1, folds = 3, device="cpu"):
        
        self._class_names = class_names
        self._device = device
        self._lambda_values = lambda_values
        self._classifier_name = classifier_name
        self._folds = folds
        self._batch_size = batch_size
        self._epochs = epochs

        self._criterion_classifier = nn.CrossEntropyLoss()
        self._metrics = metrics={'accuracy':accuracy_score}
        
        self._G_dict = self.create_generators()
        self._best_G_dict = self.create_generators()

        self._G_criterion_dict = self.create_criterion_dict()
        self._optimizers_dict = self.create_optimizers_dict()

        self._classifier = self.create_classifier()

        self._optimizer = optim.Adam(self._classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
        self._exp_lr_scheduler = lr_scheduler.StepLR(self._optimizer, step_size=5, gamma=0.1)
        self._early_stopping = EarlyStopping(patience=10, verbose=True)

        

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

    def create_optimizers_dict(self):
        optimizers_dict = {}
        
        for class_name in self._class_names:
            optimizers_dict[class_name] = optim.Adam(self._G_dict[class_name].parameters(), lr=0.0001, weight_decay=1e-4)
        
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
            print("[INFO] FREEZING")
            ct += 1
            if ct < 7:
                 for param in child.parameters():
                    param.requires_grad = False

        # Change last layer to output N classes
        num_ftrs = classifier.fc.in_features
        classifier.fc = nn.Linear(num_ftrs, len(self._class_names))
        classifier.to(self._device)

        return classifier

    def train(self, data, labels):
        
        unique, counts = np.unique(np.asarray(labels), return_counts=True)
        class_weights = np.max(counts)/counts
        class_weights /= np.max(class_weights)
        self._criterion_classifier = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights).float().to(self._device))
        print("[INFO] weights = {}".format(class_weights))

        train_size = int(0.1*len(data))
        val_size = len(data) - train_size

        my_transform = transforms.Compose( [ transforms.RandomHorizontalFlip(), transforms.RandomAffine(5), transforms.RandomRotation(5), transforms.ToTensor() ] )
        dataset = CustomTensorDataset(data, labels, transform = my_transform)
        
        #my_transform = transforms.Compose( [ transforms.RandomHorizontalFlip(), transforms.RandomAffine(5), transforms.RandomRotation(5) ] )
        #dataset = CustomTensorDataset(torch.from_numpy(data), torch.from_numpy(labels), transform = my_transform)

        #train_dataset, val_dataset = random_split(dataset, [train_size, val_size] )
                
        best_lambda = self._lambda_values[0]
        best_acc = 0.0
        best_loss = 9000000

        for lambda_class in self._lambda_values:
            print("LAMBDA: {}".format(lambda_class)) 

            best_acc_lambda = 0.0  
            best_loss_lambda = 10000         
            
            kf = StratifiedKFold(n_splits=self._folds, shuffle=True, random_state=1337)
            for fold_idx, (train_index, val_index) in enumerate(kf.split(X=np.zeros(len(data)), y=labels)):
                print("FOLD: {}".format(fold_idx))
                train_sampler = SubsetRandomSampler(train_index)
                val_sampler = SubsetRandomSampler(val_index)
                #train_loader = DataLoader(dataset=train_dataset, batch_size = self._batch_size, num_workers = 4, sampler=train_sampler)
                #val_loader = DataLoader(dataset=val_dataset, batch_size=1, num_workers=4, sampler=val_sampler)
                
                train_loader = DataLoader(dataset=dataset, batch_size = self._batch_size, num_workers = 4, sampler=train_sampler)
                val_loader = DataLoader(dataset=dataset, batch_size=1, num_workers=4, sampler=val_sampler)
                
                self._G_dict = self.create_generators()
                self._best_G_dict = self.create_generators()
                self._classifier = self.create_classifier()
                

                self._optimizer = optim.Adam(self._classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
                self._exp_lr_scheduler = lr_scheduler.StepLR(self._optimizer, step_size=5, gamma=0.1)
                self._early_stopping = EarlyStopping(patience=10, verbose=True)
                
                for epoch in range(1, self._epochs+1):
                    train_bar = tqdm(train_loader)
                    running_results = {'batch_sizes': 0, 'd_loss': 0, 'd_corrects': 0, 'd_score': 0, 'g_score': 0}

                    for class_name in self._class_names:
                        running_results['g_loss_'+class_name] = 0
                    
                    for class_name in self._class_names:
                        self._G_dict[class_name].to(self._device)
                        self._G_dict[class_name].train()
                    self._classifier.to(self._device)
                    self._classifier.train()

                    for sample, label in train_bar:
                                        
                        #sample = sample.reshape( ( sample.shape[0], sample.shape[3], sample.shape[1], sample.shape[2] ) ).float()
                                        
                        target = sample.clone().detach()
                        
                        batch_size = sample.size(0)
                        running_results['batch_sizes'] += batch_size

                        input_img = Variable(target)
                        input_img = input_img.to(self._device)
                        label = label.to(self._device)
                        
                        z = Variable(sample)
                        #z = torch.reshape(z, (z.shape[0], z.shape[3], z.shape[1], z.shape[2]))
                        #z = z.float()
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
                        loss_classifier = self._criterion_classifier(classifier_outputs, unfold_labels)

                        # Optimize classifier
                        self._optimizer.zero_grad()
                        loss_classifier.backward(retain_graph=True)
                        self._optimizer.step()

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
                                ce_toGenerator[self._class_names[current_label]] = self._criterion_classifier(class_n_outputs, class_n_labels)
                            else:
                                ce_toGenerator[class_name] = 0.0

                        # Backprop one time per generator
                        for idx, class_name in enumerate(self._class_names):
                            self._G_dict[class_name].zero_grad()
                            g_loss = self._G_criterion_dict[class_name](transformed_imgs[idx::len(self._class_names)], input_img, ce_toGenerator[class_name], float(lambda_class))
                            if idx < len(self._class_names)-1:
                                g_loss.backward(retain_graph=True)
                            else:
                                g_loss.backward()
                            self._optimizers_dict[class_name].step()

                        # Re-do computations for obtaining loss after weight updates
                        transformed_imgs = []
                        for idx, ind_label in enumerate(label):
                            for class_name in self._class_names:
                                tr_image = self._G_dict[class_name](z[idx].unsqueeze(0))[0]
                                transformed_imgs.append(tr_image)
                        transformed_imgs = torch.stack(transformed_imgs)
                        for idx, class_name in enumerate(self._class_names):
                            g_loss = self._G_criterion_dict[class_name](transformed_imgs[idx::len(self._class_names)], input_img,
                                                                    ce_toGenerator[class_name], float(lambda_class))
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
                    print("Valid Acc = {}".format(curr_acc))
                    print("Valid Loss = {}".format(valid_loss))
                    """
                    if curr_acc >= best_acc:
                        best_acc = curr_acc
                        best_lambda = lambda_class
                        self._best_G_dict = copy.deepcopy(self._G_dict)
                        self._best_classifier = copy.deepcopy(self._classifier)
                    """

                    if valid_loss <= best_loss:
                        best_loss = valid_loss
                        best_lambda = lambda_class
                        self._best_G_dict = copy.deepcopy(self._G_dict)
                        self._best_classifier = copy.deepcopy(self._classifier)
                    
                    if curr_acc >= best_acc_lambda:
                        best_acc_lambda = curr_acc
                    
                    if valid_loss < best_loss_lambda:
                        best_loss_lambda = valid_loss
                
                    self._early_stopping(valid_loss)
                    if self._early_stopping.early_stop:
                        print("Early stopping, epoch " + str(epoch))
                        break
                
            valid_loss, y_true, y_pred, val_results = self.validate(val_loader, self._best_G_dict, self._best_classifier)
            curr_acc = accuracy_score(y_true, y_pred)
            print("Summary for LAMBDA = {}".format(lambda_class))
            print("Valid Acc = {}".format(best_acc_lambda))
            print("Valid Loss = {}".format(best_loss_lambda))

        print("BEST LAMBDA: {}".format(best_lambda))
        
        self._G_dict = self._best_G_dict
        self._classifier = self._best_classifier
    

        
    def validate(self, val_loader, G_dict, classifier):
        valid_losses = []  # Store losses for each val img
        
        for class_name in self._class_names:
            G_dict[class_name].to(self._device)
        classifier.to(self._device)
            
            
        # Set models for predicting
        for class_name in self._class_names:
            G_dict[class_name].eval()
        classifier.eval()

        val_bar = tqdm(val_loader)
        val_results = {'mse': 0, 'batch_sizes': 0, 'd_corrects': 0}
        y_true = []
        y_pred = []

        # Begin inference
        for sample, label in val_bar:
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
            val_loss = self._criterion_classifier(a,b).item()
            
            valid_losses.append(val_loss)
            val_results['d_corrects'] += torch.sum(preds == label).item()
            y_true.append(label.item())
            y_pred.append(preds.item())

            val_bar.set_description(
                desc='[Validating]: Acc_D: %.4f' % (val_results['d_corrects'] / val_results['batch_sizes']))

        val_loss = np.average(np.asarray(valid_losses))
        print("Valid Loss = {}".format(val_loss))

        return val_loss, y_true, y_pred, val_results
    
    def predict(self, data):

        dataset = TensorDataset(torch.from_numpy(data))
        dataloader = DataLoader(dataset, batch_size = 1)
        
        classifier_outputs = []
        y_pred = []
        
        self._classifier.to(self._device)
        for class_name in self._class_names:
            self._G_dict[class_name].to(self._device)
        
        with torch.no_grad():
            for element in dataloader:
                sample = element[0].float()
                #sample = sample.reshape( ( sample.shape[0], sample.shape[3], sample.shape[1], sample.shape[2] ) ).float()
                sample.requires_grad = False
                sample = sample.to(self._device)
                
                a = []
                for class_name in self._class_names:
                    sr = self._G_dict[class_name](sample)
                    out = self._classifier(sr)
                    a.append(out)
                    classifier_outputs.extend(out.cpu().numpy())
                    
                total_outputs = torch.cat(a, 1).squeeze()
                max_val, idx_max = torch.max(total_outputs, 0)
                preds = idx_max % len(self._class_names)
                y_pred.append(preds.item())
        
        return np.array(classifier_outputs), np.array(y_pred)

    def evaluate(self, data, labels):
            
        classifier_outputs, y_pred = self.predict(data)
        
        with torch.no_grad():
            labels_t = []
            for i in range(len(labels)):
                labels_t.append(labels[i])
                labels_t.append(labels[i])
            labels_t = np.asarray(labels_t)

            labels_t = torch.from_numpy(labels_t)
            classifier_outputs = torch.from_numpy(classifier_outputs).float()

            classifier_outputs = classifier_outputs.to(self._device)
            labels_t = labels_t.to(self._device)
            val_loss = self._criterion_classifier(classifier_outputs, labels_t.squeeze())
            
            metrics = [val_loss.item()]
            if self._metrics is not None:
                for name, metric in self._metrics.items():
                    metrics.append(metric(y_pred, labels))

        return metrics

    def performance(self, data, labels):
        return self.evaluate(data, labels)[0]

    def get_model_params(self):
        weights = []
        
        class_weights = []
        for param in self._classifier.parameters():
            class_weights.append(param.cpu().data.numpy())
        weights.append(class_weights)
        
        for class_name in self._class_names:
            generator = self._G_dict[class_name]            
            gen_weights = []
            for param in generator.parameters():
                gen_weights.append(param.cpu().data.numpy())
            weights.append(gen_weights)

        return weights

    def set_model_params(self, params):
        with torch.no_grad():
            
            for ant, post in zip(self._classifier.parameters(), params[0]):
                ant.data = torch.from_numpy(post).float()
                
            i = 1
            for class_name in self._class_names:
                generator = self._G_dict[class_name] 
                for ant, post in zip(generator.parameters(), params[i]):
                    ant.data = torch.from_numpy(post).float()
                i = i+1
    
    def transform_data(self, data, labels, label_binarizer_1, label_binarizer_2):

        for class_name in self._class_names:
            self._G_dict[class_name]= self._G_dict[class_name].to(self._device)

        new_labels = []
        new_data = []
        for i in range(len(data)):
            sample = data[i]
            label = label_binarizer_1.inverse_transform(labels[i])[0]
            x = sample_loader(sample).to(self._device)
            for class_name in self._class_names:
                y = self._G_dict[class_name](x)
                y = y[0].cpu().detach().numpy()
                y = np.moveaxis(y, 0, -1)
                y = cv2.resize(y, dsize=(224, 224))
                new_data.append(y)
                new_label = str(label) + "T" + class_name
                new_labels.append(new_label)
        new_labels = label_binarizer_2.transform(new_labels)
    
        return np.asarray(new_data), np.asarray(new_labels)
