import torch.nn as nn
import torch
import numpy as np
from .convlstm import ConvLSTM

class Camera(nn.Module):
    ''' 
        Image pipeline implementation of the model mentioned in this paper: Deep Predictive Models for Collision Risk Assessment in Autonomous Driving
        We also reference this implementation as well: https://gitlab.com/avces/avces_tensorflow/-/blob/master/src/dpm_keras_general.py
    '''
    def __init__(self, input_shape, config):
        super(Camera, self).__init__()
        self.batch_size, self.frames, self.channels, self.height, self.width = input_shape # ex: (1, 5, 1, 64, 64)
        self.config = config

        # Hyper-parameters
        self.num_features = 8
        self.kernal_size = (5, 5)
        self.stride = (2, 2)
        self.num_layers = 1

        # Activation functions
        self.activation = nn.ReLU() if self.config.activation == 'relu' else nn.LeakyReLU(0.1) 
        self.dropout = nn.Dropout(self.config.dropout)

        # Normalization functions
        self.bnorm3d_1 = nn.BatchNorm3d(num_features=self.frames)
        self.bnorm3d_2 = nn.BatchNorm3d(num_features=self.frames)

        # Layers
        self.convlstm1 = ConvLSTM(input_dim=self.channels, hidden_dim=self.num_features, kernel_size=self.kernal_size, num_layers=self.num_layers, batch_first=True)
        self.convlstm2 = ConvLSTM(input_dim=self.num_features, hidden_dim=self.num_features, kernel_size=self.kernal_size, stride=self.stride, num_layers=self.num_layers, batch_first=True)
        self.convlstm3 = ConvLSTM(input_dim=self.num_features, hidden_dim=self.num_features, kernel_size=self.kernal_size, stride=self.stride, num_layers=self.num_layers, batch_first=True)
        self.flatten = nn.Flatten(start_dim=1)
        
    def forward(self, x):

        l1, (l1_h, l1_c) = self.convlstm1(x)
        l2, (l2_h, l2_c) = self.convlstm2(self.bnorm3d_1(l1))
        l3, (l3_h, l3_c) = self.convlstm3(self.bnorm3d_2(l2))
        #l2, (l2_h, l2_c) = self.convlstm2(l1)
        #l3, (l3_h, l3_c) = self.convlstm3(l2)
        l4 = self.flatten(l3)
        return l4

if __name__ == '__main__':
    from types import SimpleNamespace
    cfg = {'dropout': 0.1, 'device': 'cpu', 'activation': 'relu'}
    config = SimpleNamespace(**cfg)
    image = torch.rand((16, 5, 1, 64, 64))
    model = Camera(image.shape, config)

    pred = model(image)
    print(pred.shape)
