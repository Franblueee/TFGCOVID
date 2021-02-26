import numpy as np
import torch

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

    def __call__(self, val_loss, model, G_dict, path, idx=""):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            #self.save_checkpoint(val_loss, model, G_dict, path, idx)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            #self.save_checkpoint(val_loss, model, G_dict, path, idx)
            self.counter = 0

    def save_checkpoint(self, val_loss, classifier, G_dict, path, idx=""):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        for key, _ in G_dict.items():
            if idx != "":
                torch.save(G_dict[key].state_dict(), path + 'checkpoint_'+str(idx)+'_netG_'+key+'.pth')
            else:
                torch.save(G_dict[key].state_dict(), path + 'checkpoint_netG_'+key+'.pth')
        if idx != "":
            torch.save(classifier.state_dict(), path+'checkpoint_' + str(idx) + '_netD.pth')
        else:
            torch.save(classifier.state_dict(), path+'checkpoint_netD.pth')
        self.val_loss_min = val_loss
