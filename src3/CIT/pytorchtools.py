import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import ToTensor, Resize


class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, data, labels, data_size, transform=None):
        self._data = data
        self._labels = labels
        self._transform = transform
        self._data_size = data_size

    def __getitem__(self, index):
        x = self._data[index]
        y = self._labels[index]
        hr_scale = Resize(size=(self._data_size, self._data_size), interpolation=Image.BICUBIC)
        if self._transform:
            #x = Image.fromarray(self._data[index].astype(np.uint8))
            x = Image.fromarray(self._data[index])
            x = self._transform(x)
            width, height = x.size
            if width != self._data_size or height != self._data_size:
                x = hr_scale(x)

        return ToTensor()(x), ToTensor()(x), y

    def __len__(self):
        return len(self._data)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


