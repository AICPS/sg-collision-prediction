import torch.nn as nn
import torch
from .camera import Camera
import pdb

class DeepPredictiveModel(nn.Module):
    ''' 
        This baseline model is an implementation of the model mentioned in this paper: Deep Predictive Models for Collision Risk Assessment in Autonomous Driving
        We also reference this implementation as well: https://gitlab.com/avces/avces_tensorflow/-/blob/master/src/dpm_keras_general.py
    '''
    def __init__(self, input_shape, config):
        super(DeepPredictiveModel, self).__init__()

        self.cameras, self.batch_size, self.frames, self.channels, self.height, self.width = input_shape # ex: (1, 1, 5, 1, 64, 64)
        self.config = config

        # Activation functions
        self.activation = nn.ReLU() if self.config.activation == 'relu' else nn.LeakyReLU(0.1) 
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(self.config.dropout)

        # Image processing pipeline
        self.camera_models = [Camera(input_shape[1:], self.config).to(self.config.device) for i in range(self.cameras)]

        # TODO: add vehicle state information


        # TODO: Dynamically calculate in_features
        self.linear1 = nn.Linear(in_features=10240*self.cameras, out_features=64)
        self.linear2 = nn.Linear(in_features=64, out_features=2)

    def forward(self, x):
        
        # Image models
        all_cameras = torch.cat([self.camera_models[index](value) for index, value in enumerate(x)], dim=-1) #flatten

        # TODO: Vehicle state models
        l1 = self.dropout(self.activation(self.linear1(all_cameras)))
        l2 = self.softmax(self.linear2(l1))
        #l2 = self.linear2(l1)
        return l2

if __name__ == '__main__':
    from types import SimpleNamespace
    cfg = {'dropout': 0.1, 'device': 'cuda', 'activation': 'relu'}
    config = SimpleNamespace(**cfg)
    image = torch.rand((1, 32, 5, 1, 64, 64)) # (cameras, batch, time_steps, channels, height, width)
    
    # Send to GPU
    model = DeepPredictiveModel(image.shape, config).to(config.device)
    pred = model(image.to(config.device))

    print(pred.shape)
    
