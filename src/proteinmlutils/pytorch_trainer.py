import torch
import lightning.pytorch as pl
import torchmetrics
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import pandas as pd
from torch.utils.data import random_split
from torch import nn

class pytorch_model_template(pl.LightningModule):
    '''
    The template for the PyTorch model.
    '''
    def __init__(self, model_input, model_type='regression', num_heads=1, lr=1e-2, weight_decay=1e-3, weights=None, num_classes=None):
        #to do:
        #docstring
        #assert statement for dim_out that it needs to be 1 for regression and num_classes for classification.
        super().__init__()
        self.save_hyperparameters(ignore=['model_input'])
        self.model_input = model_input
        self.model_type = model_type
        self.num_classes = num_classes
        self.lr = lr
        self.num_heads = num_heads
        self.weight_decay = weight_decay
        #self.latent_vector = latent_vector
        #self.dim_out = dim_out
        self.weights = weights # Class weights (default: None)
        #TODO: Add a comment explaining the choice of loss function.
        if self.model_type == 'classification':
            if self.num_classes == 2:
                self.F1_score = torchmetrics.F1Score(num_classes=2, average='macro', task='binary')
            else:
                self.F1_score = torchmetrics.F1Score(num_classes=self.num_classes, average='macro', task='multiclass')
        elif self.model_type == 'regression':
            self.spearman_corr = torchmetrics.SpearmanCorrCoef(num_outputs=num_heads)
        else:
            raise ValueError('model_type should be either regression or classification')

    def forward(self, data):
        #TODO: Add docstring in sphinx format.
        #ToDO: should we add linear layer here or in the base_model???
        #penultimate_layer = self.model_input(data)
        logits=self.model_input(data['x'])
        #logits = self.linear_layer(penultimate_layer)
        return logits

    def training_step(self, data):
        logits = self.forward(data)
        #TO DO: don't assume that it's a classifier
        
        #y_pred = torch.argmax(logits, dim=1)
        loss = self.loss_fn(logits, data)
        
        if self.model_type == 'regression': #todo: move it inside loss function.
            dict_entry={"train_loss": loss, "train_spearman_corr":torch.mean(self.spearman_corr(logits, data['y'])), "manual_epoch": self.current_epoch}
            #print(dict_entry)
        elif self.model_type == 'classification':
            raise NotImplementedError("Classification not implemented yet")
            #dict_entry={"train_loss": loss, "train_F1_score":self.F1_score(y_pred, data['y'].long()), "manual_epoch": self.current_epoch}
        
        else:
            raise ValueError('model_type should be either regression or classification')
        
        self.log_dict(dict_entry, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, data, batch_idx=None):
        #Note that batch_idx is needed for Pytorch Lightning syntax, but it is not used in this function.
        # this is the validation loop
        logits = self.forward(data)
        #y_pred = torch.argmax(logits, dim=1)
        val_loss = self.loss_fn(logits, data)
        # Logging to TensorBoard (if installed) by default
        if self.model_type == 'regression':
            dict_entry={"val_loss": val_loss, "val_spearman_corr":torch.mean(self.spearman_corr(logits, data['y'])), "manual_epoch": self.current_epoch}
        elif self.model_type == 'classification':
            raise NotImplementedError("Classification not implemented yet")
        else:
            raise ValueError('model_type should be either regression or classification')    
        self.log_dict(dict_entry, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        
        return val_loss

    def test_step(self, data):
        # this is the test loop
        logits = self.forward(data)
        #y_pred = torch.argmax(logits, dim=1)
        test_loss = self.loss_fn(logits, data)
        # Logging to TensorBoard (if installed) by default
        if self.model_type == 'regression':
            dict_entry={"test_loss": test_loss, "test_spearman_corr":torch.mean(self.spearman_corr(logits, data['y'])), "manual_epoch": self.current_epoch}
        elif self.model_type == 'classification':
            raise NotImplementedError("Classification not implemented yet")
        else:
            raise ValueError('model_type should be either regression or classification')    
        self.log_dict(dict_entry, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)

        return test_loss


    def configure_optimizers(self):
        '''
        Input:
        None
        Output:
        optimizer: The optimizer to be used
        '''
        #TODO: Why not self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)?
        #I am not sure if it would work. 
        #Potential bug: It only updates the parameters of the model_input, not the linear layer.
        optimizer = optim.Adadelta(self.model_input.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
    
    def loss_fn(self, y_pred, data):
        '''
        Input:
        y_pred: The predicted values
        data: The input data
        Output:
        loss: The loss function
        '''
        if self.model_type == 'regression':
            for head_i in range(self.num_heads):
                if head_i == 0:
                    loss = nn.functional.mse_loss(y_pred[:,head_i], data["y"][:,head_i])
                else:
                    loss += nn.functional.mse_loss(y_pred[:,head_i], data["y"][:,head_i])
                    #print(f"Head {head_i} loss: {nn.functional.mse_loss(y_pred[:,head_i], data['y'][:,head_i])}")
            return loss
        elif self.model_type == 'classification':
            return nn.functional.cross_entropy(y_pred, data['y'].long(), reduction='sum', weight=self.weights)
        else:
            raise ValueError('model_type should be either regression or classification')





class DictTensorDataset(Dataset):
    '''
    Custom Dataset for loading protein sequences and their labels
    in a dictionary format.
    X: A tensor or numpy array
    y: A tensor or numpy array
    type: "numpy" or "torch" indicating the type of input data
    Output:
    A dictionary with keys 'x' and 'y' for each sample.
    '''
    def __init__(self, X, y, type="numpy"):
        
        
        #convert numpy arrays to torch tensors if they are not already
        if type == "numpy":
            self.X = torch.from_numpy(X).float()
            self.y=torch.from_numpy(y).float()
        elif type == "torch":
            #raise warning if the tensors are not float
            if self.X.dtype != torch.float32:
                print("Warning: X is not float32, converting to float32")
                self.X = self.X.float()
            else:
                self.X = self.X
            if self.y.dtype != torch.float32:
                print("Warning: y is not float32, converting to float32")
                self.y = self.y.float()
            else:
                self.y = self.y
        else:
            raise ValueError("type should be either numpy or torch")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        
        return {"x": self.X[idx], "y": self.y[idx]}
