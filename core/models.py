import torch
import torch.nn as nn
import DPM.dpm as dpm
from mrgcn import RGCNSAGPooling, MRGCN, MRGIN

class BaselineModel(nn.Module):
    ''' 
        This baseline model is an implementation of the model mentioned in this paper: Deep Predictive Models for Collision Risk Assessment in Autonomous Driving
        We also reference this implementation as well: https://gitlab.com/avces/avces_tensorflow/-/blob/master/src/dpm_keras_general.py
    '''
    def __init__(self, input_shape):
        super(BaselineModel, self).__init__()
        pass

    def forward(self, x):
        pass


#TODO: refactor to incorporate MRGCN w/ new temporal component
class OurModel(nn.Module):
    ''' 
        Our model will basically utilize graph-related architecture to process prediction problem.
    '''
    def __init__(self):
        super(OurModel, self).__init__()
        pass
    
    def forward(self, X):
        pass