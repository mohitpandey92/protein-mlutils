import torch
import lightning.pytorch as pl
import torchmetrics
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import pandas as pd
from torch.utils.data import random_split
from torch import nn


class CNN_FNN_model(nn.Module):
    # Good CNN visulization
    #https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    def __init__(
            self, n_classes: int, protein_length: int,
            cnn_outchannel_1: int = 32, cnn_outchannel_2: int = 32, 
            linear_features_input: int=200, linear_bottleneck_dim: int=100,
            dropout_rate: float = 0.45, kernel_size: int=300, L_input: int=100, 
            stride: int=1, **kwargs
    ):
        super(CNN_FNN_model, self).__init__()
        self.CNN_first_layer =nn.Conv1d(int(protein_length/2), cnn_outchannel_1, kernel_size=kernel_size,stride=stride)
        self.CNN_second_layer = nn.Conv1d(int(cnn_outchannel_1/2), cnn_outchannel_2, kernel_size=kernel_size,stride=1)
        self.maxpool_first_layer = nn.MaxPool2d(2)
        #self.fcc_layer=nn.Linear(linear_features_input, linear_bottleneck_dim)
        self.linear_layer=nn.Linear(linear_features_input, linear_bottleneck_dim)
        #self.second_layer=nn.Linear(linear_bottleneck_dim, linear_bottleneck_dim)
        self.outputlayer=nn.Linear(linear_bottleneck_dim, n_classes)
        self.leaky_relu= nn.LeakyReLU()
        self.fcc_classifier = nn.Sequential(
            #self.linear_layer,  # n_features x 32
            self.linear_layer,
            self.leaky_relu,
            nn.Dropout(dropout_rate),
            #self.second_layer,  # n_features x 32
            #self.leaky_relu,
            #nn.Dropout(dropout_rate),
            #self.second_layer,  # n_features x 32
            #self.leaky_relu,
            #nn.Dropout(dropout_rate),
            #self.second_layer,  # n_features x 32
            #self.leaky_relu,
            #nn.Dropout(dropout_rate),
            self.outputlayer
        )
    def forward(self, x):
        """
            L = protein length
            B = batch-size
            F = number of features (1024 for embeddings)
            N = number of classes (9 for conservation)
        """
        # IN: X = (B x L x F)
        
        #x=self.maxpool_first_layer(x)
        #print("after maxpooling", x.shape)
        x=self.leaky_relu(self.CNN_first_layer(x))
        #print("after conv1d", x.shape)
        x=self.maxpool_first_layer(x)
        #print("after maxpool", x.shape)
        x=self.leaky_relu(self.CNN_second_layer(x))
        #print("after conv1d", x.shape)
        x=torch.flatten(x, start_dim=1)
        #print(x.shape)
        Yhat = self.fcc_classifier(x)  # OUT: Yhat_consurf = (B x L x N)
        return Yhat
    
    
def size_linear_features_input_after_CNN(input_feature_size,kernel_size, stride, cnn_outchannel_2):
    '''
    Note that this is based on given architecture choice: MaxPool --> Conv --> MaxPool-->Conv --> FC
    '''
    L_input=input_feature_size
    L_out=int((L_input/2))
    L_input=L_out
    
    #print("L_out after max features", L_out)
    
    L_out=int((L_input+kernel_size-2)/stride)
    L_input=L_out
    
    #print("L_out after conv1d", L_out)
    
    
    #maxpool
    L_out=int((L_input/2))
    L_input=L_out
    #print("L_out after second max", L_out)
    
            
    L_out=int((L_input+kernel_size-2))
    L_out=L_out -1 # I don't know what is the reason for minus 1
    #print("L_out after conv1d second layer", L_out)
    
    #kernel_size=4
    linear_features_input=L_out*cnn_outchannel_2
    return  linear_features_input
