import torch
import torch.nn as nn
import torch.nn.functional as F
from torchnlp.nn import Attention
from torch.nn import Linear, LSTM
from torch_geometric.nn import RGCNConv, TopKPooling, FastRGCNConv
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from torch_geometric.nn.pool.topk_pool import topk, filter_adj
from torch_geometric.utils import softmax
import pdb


class RGCNSAGPooling(torch.nn.Module):
    def __init__(self, in_channels, num_relations, ratio=0.5, min_score=None,
                 multiplier=1, nonlinearity=torch.tanh, rgcn_func="FastRGCNConv", **kwargs):
        super(RGCNSAGPooling, self).__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.gnn = FastRGCNConv(in_channels, 1, num_relations, **kwargs) if rgcn_func=="FastRGCNConv" else RGCNConv(in_channels, 1, num_relations, **kwargs)
        self.min_score = min_score
        self.multiplier = multiplier
        self.nonlinearity = nonlinearity

        self.reset_parameters()

    def reset_parameters(self):
        self.gnn.reset_parameters()


    def forward(self, x, edge_index, edge_attr=None, batch=None, attn=None):
        """"""
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        attn = x if attn is None else attn
        attn = attn.unsqueeze(-1) if attn.dim() == 1 else attn
        score = self.gnn(attn, edge_index, edge_attr).view(-1)

        if self.min_score is None:
            score = self.nonlinearity(score)
        else:
            score = softmax(score, batch)

        perm = topk(score, self.ratio, batch, self.min_score)
        x = x[perm] * score[perm].view(-1, 1)
        x = self.multiplier * x if self.multiplier != 1 else x

        batch = batch[perm]
        edge_index, edge_attr = filter_adj(edge_index, edge_attr, perm,
                                           num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm, score[perm]


    def __repr__(self):
        return '{}({}, {}, {}={}, multiplier={})'.format(
            self.__class__.__name__, self.gnn.__class__.__name__,
            self.in_channels,
            'ratio' if self.min_score is None else 'min_score',
            self.ratio if self.min_score is None else self.min_score,
            self.multiplier)

class MRGCN(nn.Module):
    
    def __init__(self, config):
        super(MRGCN, self).__init__()

        self.num_features = config.num_features
        self.num_relations = config.num_relations
        self.num_classes  = config.nclass
        self.num_layers = config.num_layers #defines number of RGCN conv layers.
        self.hidden_dim = config.hidden_dim
        self.layer_spec = None if config.layer_spec == None else list(map(int, config.layer_spec.split(',')))
        if self.layer_spec != None and self.num_layers != len(self.layer_spec):
            raise ValueError("num_layers does not match the length of layer_spec") #we want this to break here because our data logging will not be accurate if config.num_layers != len(config.layer_spec)
        self.lstm_dim1 = config.lstm_input_dim
        self.lstm_dim2 = config.lstm_output_dim
        self.rgcn_func = FastRGCNConv if config.conv_type == "FastRGCNConv" else RGCNConv
        self.activation = F.relu if config.activation == 'relu' else F.leaky_relu
        self.pooling_type = config.pooling_type
        self.readout_type = config.readout_type
        self.temporal_type = config.temporal_type
        self.lstm_layers = config.lstm_layers #defines number of lstm layers
        self.dropout = config.dropout
        self.conv = []
        total_dim = 0

        if self.layer_spec == None:
            if self.num_layers > 0:
                self.conv.append(self.rgcn_func(self.num_features, self.hidden_dim, self.num_relations).to(config.device))
                total_dim += self.hidden_dim
                for i in range(1, self.num_layers):
                    self.conv.append(self.rgcn_func(self.hidden_dim, self.hidden_dim, self.num_relations).to(config.device))
                    total_dim += self.hidden_dim
            else:
                self.fc0_5 = Linear(self.num_features, self.hidden_dim)
                total_dim += self.hidden_dim
        else:
            if self.num_layers > 0:
                print("using layer specification and ignoring hidden_dim parameter.")
                print("layer_spec: " + str(self.layer_spec))
                self.conv.append(self.rgcn_func(self.num_features, self.layer_spec[0], self.num_relations).to(config.device))
                total_dim += self.layer_spec[0]
                for i in range(1, self.num_layers):
                    self.conv.append(self.rgcn_func(self.layer_spec[i-1], self.layer_spec[i], self.num_relations).to(config.device))
                    total_dim += self.layer_spec[i]

            else:
                self.fc0_5 = Linear(self.num_features, self.hidden_dim)
                total_dim += self.hidden_dim

        if self.pooling_type == "sagpool":
            self.pool1 = RGCNSAGPooling(total_dim, self.num_relations, ratio=config.pooling_ratio, rgcn_func=config.conv_type)
        elif self.pooling_type == "topk":
            self.pool1 = TopKPooling(total_dim, ratio=config.pooling_ratio)

        self.fc1 = Linear(total_dim, self.lstm_dim1)
        
        if "lstm" in self.temporal_type:
            self.lstm = LSTM(self.lstm_dim1, self.lstm_dim2, batch_first=True, num_layers=config.lstm_layers)
            self.attn = Attention(self.lstm_dim2)
            self.lstm_decoder = LSTM(self.lstm_dim2, self.lstm_dim2, batch_first=True)
        else:
            self.fc1_5 = Linear(self.lstm_dim1, self.lstm_dim2)

        self.fc2 = Linear(self.lstm_dim2, self.num_classes)


    def forward(self, x, edge_index, edge_attr, batch=None):
        attn_weights = dict()
        outputs = []
        if self.num_layers > 0:
            for i in range(self.num_layers):
                x = self.activation(self.conv[i](x, edge_index, edge_attr))
                x = F.dropout(x, self.dropout, training=self.training)
                outputs.append(x)
            x = torch.cat(outputs, dim=-1)
        else:
            x = self.activation(self.fc0_5(x))

        if self.pooling_type == "sagpool":
            x, edge_index, _, attn_weights['batch'], _, _ = self.pool1(x, edge_index, edge_attr=edge_attr, batch=batch)
        elif self.pooling_type == "topk":
            x, edge_index, _, attn_weights['batch'], attn_weights['pool_perm'], attn_weights['pool_score'] = self.pool1(x, edge_index, edge_attr=edge_attr, batch=batch)
        else: 
            attn_weights['batch'] = batch

        if self.readout_type == "add":
            x = global_add_pool(x, attn_weights['batch'])
        elif self.readout_type == "mean":
            x = global_mean_pool(x, attn_weights['batch'])
        elif self.readout_type == "max":
            x = global_max_pool(x, attn_weights['batch'])
        else:
            pass

        x = self.activation(self.fc1(x))
    
        if self.temporal_type == "mean":
            x = self.activation(self.fc1_5(x.mean(axis=0)))
        elif self.temporal_type == "lstm_last":
            x_predicted, (h, c) = self.lstm(x.unsqueeze(0))
            x = h.flatten()
        elif self.temporal_type == "lstm_sum":
            x_predicted, (h, c) = self.lstm(x.unsqueeze(0))
            x = x_predicted.sum(dim=1).flatten()
        elif self.temporal_type == "lstm_attn":
            x_predicted, (h, c) = self.lstm(x.unsqueeze(0))
            x, attn_weights['lstm_attn_weights'] = self.attn(h.view(1,1,-1), x_predicted)
            x, (h_decoder, c_decoder) = self.lstm_decoder(x, (h, c))
            x = x.flatten()
        elif self.temporal_type == "lstm_seq": #used for step-by-step sequence prediction. 
            x_predicted, (h, c) = self.lstm(x.unsqueeze(0)) #x_predicted is sequence of predictions for each frame, h is hidden state of last item, c is last cell state
            x = x_predicted.squeeze(0) #we return x_predicted as we want to know the output of the LSTM for each value in the sequence
        elif self.temporal_type == 'none': #this option uses no temporal modeling at all. 
            x = self.activation(self.fc1_5(x))
        else:
            pass

        return F.log_softmax(self.fc2(x), dim=-1), attn_weights




    
#implementation of MRGCN using a GIN style readout.
class MRGIN(nn.Module):
    def __init__(self, config):
        super(MRGIN, self).__init__()
        self.num_features = config.num_features
        self.num_relations = config.num_relations
        self.num_classes  = config.nclass
        self.num_layers = config.num_layers #defines number of RGCN conv layers.
        self.hidden_dim = config.hidden_dim
        self.layer_spec = None if config.layer_spec == None else list(map(int, config.layer_spec.split(',')))
        self.lstm_dim1 = config.lstm_input_dim
        self.lstm_dim2 = config.lstm_output_dim
        self.rgcn_func = FastRGCNConv if config.conv_type == "FastRGCNConv" else RGCNConv
        self.activation = F.relu if config.activation == 'relu' else F.leaky_relu
        self.pooling_type = config.pooling_type
        self.readout_type = config.readout_type
        self.temporal_type = config.temporal_type
        self.dropout = config.dropout
        self.conv = []
        self.pool = []
        total_dim = 0

        if self.layer_spec == None:
            for i in range(self.num_layers):
                if i == 0:
                    self.conv.append(self.rgcn_func(self.num_features, self.hidden_dim, self.num_relations).to(config.device))
                else:
                    self.conv.append(self.rgcn_func(self.hidden_dim, self.hidden_dim, self.num_relations).to(config.device))
                if self.pooling_type == "sagpool":
                    self.pool.append(RGCNSAGPooling(self.hidden_dim, self.num_relations, ratio=config.pooling_ratio, rgcn_func=config.conv_type).to(config.device))
                elif self.pooling_type == "topk":
                    self.pool.append(TopKPooling(self.hidden_dim, ratio=config.pooling_ratio).to(config.device))
                total_dim += self.hidden_dim
        
        else:
            print("using layer specification and ignoring hidden_dim parameter.")
            print("layer_spec: " + str(self.layer_spec))
            for i in range(self.num_layers):
                if i == 0:
                    self.conv.append(self.rgcn_func(self.num_features, self.layer_spec[0], self.num_relations).to(config.device))
                else:
                    self.conv.append(self.rgcn_func(self.layer_spec[i-1], self.layer_spec[i], self.num_relations).to(config.device))
                if self.pooling_type == "sagpool":
                    self.pool.append(RGCNSAGPooling(self.layer_spec[i], self.num_relations, ratio=config.pooling_ratio, rgcn_func=config.conv_type).to(config.device))
                elif self.pooling_type == "topk":
                    self.pool.append(TopKPooling(self.layer_spec[i], ratio=config.pooling_ratio).to(config.device))
                total_dim += self.layer_spec[i]
            
        self.fc1 = Linear(total_dim, self.lstm_dim1)
        
        if "lstm" in self.temporal_type:
            self.lstm = LSTM(self.lstm_dim1, self.lstm_dim2, batch_first=True)
            self.attn = Attention(self.lstm_dim2)
        
        self.fc2 = Linear(self.lstm_dim2, self.num_classes)



    def forward(self, x, edge_index, edge_attr, batch=None):
        attn_weights = dict()
        outputs = []

        #readout performed after each layer and concatenated
        for i in range(self.num_layers):
            x = self.activation(self.conv[i](x, edge_index, edge_attr))
            x = F.dropout(x, self.dropout, training=self.training)
            if self.pooling_type == "sagpool":
                p, _, _, batch2, attn_weights['pool_perm'], attn_weights['pool_score'] = self.pool[i](x, edge_index, edge_attr=edge_attr, batch=batch)
            elif self.pooling_type == "topk":
                p, _, _, batch2, attn_weights['pool_perm'], attn_weights['pool_score'] = self.pool[i](x, edge_index, edge_attr=edge_attr, batch=batch)
            else:
                p = x
                batch2 = batch
            if self.readout_type == "add":
                r = global_add_pool(p, batch2)
            elif self.readout_type == "mean":
                r = global_mean_pool(p, batch2)
            elif self.readout_type == "max":
                r = global_max_pool(p, batch2)
            else:
                r = p
            outputs.append(r)

        x = torch.cat(outputs, dim=-1)
        x = self.activation(self.fc1(x))

        if self.temporal_type == "mean":
            x = self.activation(x.mean(axis=0))
        elif self.temporal_type == "lstm_last":
            x_predicted, (h, c) = self.lstm(x.unsqueeze(0))
            x = h.flatten()
        elif self.temporal_type == "lstm_sum":
            x_predicted, (h, c) = self.lstm(x.unsqueeze(0))
            x = x_predicted.sum(dim=1).flatten()
        elif self.temporal_type == "lstm_attn":
            x_predicted, (h, c) = self.lstm(x.unsqueeze(0))
            x, attn_weights['lstm_attn_weights'] = self.attn(h.view(1,1,-1), x_predicted)
            x = x.flatten()
        elif self.temporal_type == "lstm_seq": #used for step-by-step sequence prediction. 
            x_predicted, (h, c) = self.lstm(x.unsqueeze(0)) #x_predicted is sequence of predictions for each frame, h is hidden state of last item, c is last cell state
            x = x_predicted.squeeze(0) #we return x_predicted as we want to know the output of the LSTM for each value in the sequence
        else:
            pass
                
        return F.log_softmax(self.fc2(x), dim=-1), attn_weights