import os, sys, pdb
sys.path.append(os.path.dirname(sys.path[0]))
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import random
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn import preprocessing
from matplotlib import pyplot as plt

from core.relation_extractor import Relations
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
from core.mrgcn import *
from torch_geometric.data import Data, DataLoader, DataListLoader
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.utils import resample
import pickle as pkl
from sklearn.model_selection import train_test_split, StratifiedKFold
from core.metrics import *

from collections import Counter
import wandb

class Config:
    '''Argument Parser for script to train scenegraphs.'''
    def __init__(self, args):
        self.parser = ArgumentParser(description='The parameters for training the scene graph using GCN.')
        self.parser.add_argument('--cache_path', type=str, default="../script/image_dataset.pkl", help="Path to the cache file.")
        self.parser.add_argument('--transfer_path', type=str, default="", help="Path to the transfer file.")
        self.parser.add_argument('--model_load_path', type=str, default="./model/model_best_val_loss_.vec.pt", help="Path to load cached model file.")
        self.parser.add_argument('--model_save_path', type=str, default="./model/model_best_val_loss_.vec.pt", help="Path to save model file.")
        self.parser.add_argument('--split_ratio', type=float, default=0.3, help="Ratio of dataset withheld for testing.")
        self.parser.add_argument('--downsample', type=lambda x: (str(x).lower() == 'true'), default=False, help='Set to true to downsample dataset.')
        self.parser.add_argument('--learning_rate', default=0.00005, type=float, help='The initial learning rate for GCN.')
        self.parser.add_argument('--n_folds', type=int, default=1, help='Number of folds for cross validation')
        self.parser.add_argument('--seed', type=int, default=0, help='Random seed.')
        self.parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
        self.parser.add_argument('--activation', type=str, default='relu', help='Activation function to use, options: [relu, leaky_relu].')
        self.parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
        self.parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
        self.parser.add_argument('--nclass', type=int, default=2, help="The number of classes for dynamic graph classification (currently only supports 2).")
        self.parser.add_argument('--batch_size', type=int, default=16, help='Number of graphs in a batch.')
        self.parser.add_argument('--device', type=str, default="cuda", help='The device on which models are run, options: [cuda, cpu].')
        self.parser.add_argument('--test_step', type=int, default=5, help='Number of training epochs before testing the model.')
        self.parser.add_argument('--inference_mode', type=str, default="5_frames", help='Window size of frames before making one prediction (all_frames for per-frame prediction).')
        self.parser.add_argument('--model', type=str, default="mrgcn", help="Model to be used intrinsically. options: [mrgcn, mrgin]")
        self.parser.add_argument('--conv_type', type=str, default="FastRGCNConv", help="type of RGCNConv to use [RGCNConv, FastRGCNConv].")
        self.parser.add_argument('--num_layers', type=int, default=2, help="Number of layers in the network.")
        self.parser.add_argument('--hidden_dim', type=int, default=64, help="Hidden dimension in RGCN.")
        self.parser.add_argument('--layer_spec', type=str, default=None, help="manually specify the size of each layer in format l1,l2,l3 (no spaces).")
        self.parser.add_argument('--pooling_type', type=str, default="sagpool", help="Graph pooling type, options: [sagpool, topk, None].")
        self.parser.add_argument('--pooling_ratio', type=float, default=0.5, help="Graph pooling ratio.")        
        self.parser.add_argument('--readout_type', type=str, default="add", help="Readout type, options: [max, mean, add].")
        self.parser.add_argument('--temporal_type', type=str, default="lstm_seq", help="Temporal type, options: [mean, lstm_last, lstm_sum, lstm_attn, lstm_seq].")
        self.parser.add_argument('--lstm_input_dim', type=int, default=50, help="LSTM input dimensions.")
        self.parser.add_argument('--lstm_output_dim', type=int, default=20, help="LSTM output dimensions.")
        self.parser.add_argument('--lstm_layers', type=int, default=1, help="LSTM layers.")

        args_parsed = self.parser.parse_args(args)
        wandb.init(project="av-scenegraph")
        wandb_config = wandb.config
        
        for arg_name in vars(args_parsed):
            self.__dict__[arg_name] = getattr(args_parsed, arg_name)
            wandb_config[arg_name] = getattr(args_parsed, arg_name)
            
        self.cache_path = Path(self.cache_path).resolve()
        self.transfer_path = Path(self.transfer_path).resolve() if self.transfer_path != "" else None

def build_scenegraph_dataset(cache_path, train_to_test_ratio=0.3, downsample=False, seed=0, transfer_path=None):
    '''
    Dataset format 
        scenegraphs_sequence: dict_keys(['sequence', 'label', 'folder_name'])
            'sequence': scenegraph metadata
            'label': classification output [0 -> non_risky (negative), 1 -> risky (positive)]
            'folder_name': foldername storing sequence data

    Dataset modes
        no downsample
            all sequences used for the train and test set regardless of class distribution
        downsample  
            equal amount of positive and negative sequences used for the train and test set
        transfer 
            replaces original test set with another dataset 
    '''
    dataset_file = open(cache_path, "rb")
    scenegraphs_sequence, feature_list = pkl.load(dataset_file)

    class_0 = []
    class_1 = []

    for g in scenegraphs_sequence:
        if g['label'] == 0:
            class_0.append(g)
        elif g['label'] == 1:
            class_1.append(g)
        
    y_0 = [0]*len(class_0)
    y_1 = [1]*len(class_1)
    min_number = min(len(class_0), len(class_1))
    
    # dataset class distribution
    modified_class_0, modified_y_0 = resample(class_0, y_0, n_samples=min_number) if downsample else class_0, y_0
    train, test, _, _ = train_test_split(modified_class_0+class_1, modified_y_0+y_1, test_size=train_to_test_ratio, shuffle=True, stratify=modified_y_0+y_1, random_state=seed)
    
    # transfer learning
    if transfer_path != None:
        train = np.append(train, test, axis=0)
        test, _ = pkl.load(open(transfer_path, "rb"))

    return train, test, feature_list


class DynKGTrainer:

    def __init__(self, args):
        self.config = Config(args)
        self.args = args
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        if not self.config.cache_path.exists():
            raise Exception("The cache file does not exist.")

        if not self.config.temporal_type in ["lstm_seq", 'none']:
            raise NotImplementedError("This version of dynkg_trainer does not support temporal types other than step-by-step sequence prediction (lstm_seq) or 'none'.")

        self.best_val_loss = 99999
        self.best_epoch = 0
        self.best_val_acc = 0
        self.best_val_auc = 0
        self.best_val_confusion = []
        self.best_val_f1 = 0
        self.best_val_mcc = -1.0
        self.best_val_acc_balanced = 0
        self.best_avg_pred_frame = 0
        self.log = False


    def split_dataset(self):
        self.training_data, self.testing_data, self.feature_list = build_scenegraph_dataset(self.config.cache_path, self.config.split_ratio, downsample=self.config.downsample, seed=self.config.seed, transfer_path=self.config.transfer_path)
        total_train_labels = np.concatenate([np.full(len(data['sequence']), data['label']) for data in self.training_data]) # used to compute frame-level class weighting
        total_test_labels  = np.concatenate([np.full(len(data['sequence']), data['label']) for data in self.testing_data])
        self.training_labels = [data['label'] for data in self.training_data]
        self.testing_labels  = [data['label'] for data in self.testing_data]
        self.class_weights = torch.from_numpy(compute_class_weight('balanced', np.unique(total_train_labels), total_train_labels))
        if self.config.n_folds <= 1:
            print("Number of Training Sequences Included: ", len(self.training_data))
            print("Number of Testing Sequences Included: ", len(self.testing_data))
            print("Number of Training Labels in Each Class: " + str(np.unique(total_train_labels, return_counts=True)[1]) + ", Class Weights: " + str(self.class_weights))
            print("Number of Testing Labels in Each Class: " + str(np.unique(total_test_labels, return_counts=True)[1]) + ", Class Weights: " + str(self.class_weights))


    def build_model(self):
        self.config.num_features = len(self.feature_list)
        self.config.num_relations = max([r.value for r in Relations])+1
        if self.config.model == "mrgcn":
            self.model = MRGCN(self.config).to(self.config.device)
        elif self.config.model == "mrgin":
            self.model = MRGIN(self.config).to(self.config.device)
        else:
            raise Exception("model selection is invalid: " + self.config.model)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        if self.class_weights.shape[0] < 2:
            self.loss_func = nn.CrossEntropyLoss()
        else:
            self.loss_func = nn.CrossEntropyLoss(weight=self.class_weights.float().to(self.config.device))

        wandb.watch(self.model, log="all")


    # Pick between Standard Training and KFold Cross Validation Training
    def learn(self):
        if self.config.n_folds <= 1 or self.config.transfer_path != None:
            print('\nRunning Standard Training Loop\n')
            self.train()
        else:
            print('\nRunning {}-Fold Cross Validation Training Loop\n'.format(self.config.n_folds))
            self.cross_valid()


    def cross_valid(self):

        # KFold cross validation with similar class distribution in each fold
        skf = StratifiedKFold(n_splits=self.config.n_folds)
        X = np.array(self.training_data + self.testing_data)
        y = np.array(self.training_labels + self.testing_labels)

        # self.results stores average metrics for the the n_folds
        self.results = {}
        self.fold = 1

        # Split training and testing data based on n_splits (Folds)
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            self.training_data = X_train
            self.testing_data  = X_test
            self.training_labels = y_train
            self.testing_labels  = y_test

            # To compute frame-level class weighting
            total_train_labels = np.concatenate([np.full(len(data['sequence']), data['label']) for data in self.training_data]) 
            total_test_labels = np.concatenate([np.full(len(data['sequence']), data['label']) for data in self.testing_data])
            self.class_weights = torch.from_numpy(compute_class_weight('balanced', np.unique(total_train_labels), total_train_labels))
            
            print('\nFold {}'.format(self.fold))
            print("Number of Training Sequences Included: ", len(self.training_data))
            print("Number of Testing Sequences Included: ",  len(self.testing_data))
            print("Number of Training Labels in Each Class: " + str(np.unique(total_train_labels, return_counts=True)[1]) + ", Class Weights: " + str(self.class_weights))
            print("Number of Testing Labels in Each Class: " + str(np.unique(total_test_labels, return_counts=True)[1]) + ", Class Weights: " + str(self.class_weights))
            
            self.best_val_loss = 99999
            self.train()
            self.log = True
            outputs_train, labels_train, outputs_test, labels_test, metrics = self.evaluate(self.fold)
            self.update_cross_valid_metrics(outputs_train, labels_train, outputs_test, labels_test, metrics)
            self.log = False

            if self.fold != self.config.n_folds:            
                del self.model
                del self.optimizer
                self.build_model()
                
            self.fold += 1            
        del self.results


    def train(self):
        tqdm_bar = tqdm(range(self.config.epochs))

        for epoch_idx in tqdm_bar: # iterate through epoch   
            acc_loss_train = 0
            self.sequence_loader = DataListLoader(self.training_data, batch_size=self.config.batch_size)

            for data_list in self.sequence_loader: # iterate through batches of the dataset
                self.model.train()
                self.optimizer.zero_grad()
                labels = torch.empty(0).long().to(self.config.device)
                outputs = torch.empty(0,2).to(self.config.device)

                for sequence in data_list: # iterate through scene-graph sequences in the batch
                    data, label = sequence['sequence'], sequence['label']
                    graph_list = [Data(x=g['node_features'], edge_index=g['edge_index'], edge_attr=g['edge_attr']) for g in data]
                    self.train_loader = DataLoader(graph_list, batch_size=len(graph_list))
                    sequence = next(iter(self.train_loader)).to(self.config.device)
                    output, _ = self.model.forward(sequence.x, sequence.edge_index, sequence.edge_attr, sequence.batch)
                    label = torch.LongTensor(np.full(output.shape[0], label)).to(self.config.device) #fill label to length of the sequence. shape (len_input_sequence, 1)
                    labels  = torch.cat([labels, label], dim=0)
                    outputs = torch.cat([outputs, output.view(-1, 2)], dim=0) #in this case the output is of shape (len_input_sequence, 2)

                loss_train = self.loss_func(outputs, labels)
                loss_train.backward()
                acc_loss_train += loss_train.detach().cpu().item() * len(data_list)
                self.optimizer.step()
                del loss_train

            acc_loss_train /= len(self.training_data)
            tqdm_bar.set_description('Epoch: {:04d}, loss_train: {:.4f}'.format(epoch_idx, acc_loss_train))

            if epoch_idx % self.config.test_step == 0:
                self.evaluate(epoch_idx)

    def inference(self, testing_data, testing_labels, mode='5_frames'): # change mode='all_frames' to run per-frame prediction
        labels = torch.LongTensor().to(self.config.device)
        outputs = torch.FloatTensor().to(self.config.device)
        acc_loss_test = 0
        attns_weights = []
        node_attns = []
        sum_prediction_frame = 0
        sum_seq_len = 0
        num_risky_sequences = 0
        num_safe_sequences = 0
        sum_predicted_risky_indices = 0 #sum is calculated as (value * (index+1))/sum(range(seq_len)) for each value and index in the sequence.
        sum_predicted_safe_indices = 0  #sum is calculated as ((1-value) * (index+1))/sum(range(seq_len)) for each value and index in the sequence.
        inference_time = 0
        prof_result = ""
        correct_risky_seq = 0
        correct_safe_seq = 0
        incorrect_risky_seq = 0
        incorrect_safe_seq = 0
        num_sequences = 0

        with torch.autograd.profiler.profile(enabled=False, use_cuda=True) as prof:
            with torch.no_grad():
                for i in range(len(testing_data)): # iterate through sequences of scenegraphs

                    # determine number of frames per clip and amount of frames to evaluate
                    frames_per_clip = len(testing_data[i]['sequence'])
                    frames_to_evaluate = mode.split('_')[0]
                    if frames_to_evaluate.isdigit():
                        frames_to_evaluate = int(frames_to_evaluate)
                    else:
                        frames_to_evaluate = frames_per_clip
                    
                    pred_all = frames_to_evaluate == frames_per_clip # determine to use all outputs (True) or last output of lstm (False)
                        
                    # run model inference
                    for j in range(frames_per_clip - frames_to_evaluate + 1):
                        data, label = testing_data[i]['sequence'][j:j+frames_to_evaluate], testing_labels[i]
                        data_list = [Data(x=g['node_features'], edge_index=g['edge_index'], edge_attr=g['edge_attr']) for g in data]
                        self.test_loader = DataLoader(data_list, batch_size=len(data_list))
                        sequence = next(iter(self.test_loader)).to(self.config.device)
                        self.model.eval()
                        #start = torch.cuda.Event(enable_timing=True)
                        #end =  torch.cuda.Event(enable_timing=True)
                        #start.record()
                        output, attns = self.model.forward(sequence.x, sequence.edge_index, sequence.edge_attr, sequence.batch)
                        #end.record()
                        #torch.cuda.synchronize()
                        inference_time += 0#start.elapsed_time(end)
                        output = output.view(-1,2)
                        seq_len = output.shape[0]
                        label = torch.LongTensor(np.full(seq_len, label)).to(self.config.device) #fill label to length of the sequence.
                        

                        if not pred_all:
                            # currently not supporting the attention weights when mode != 'all_frames' or pred_all == False
                            output = output[-1].unsqueeze(dim=0)
                            label = label[-1].unsqueeze(dim=0)
                            
                        outputs = torch.cat([outputs, output], dim=0)
                        labels = torch.cat([labels, label], dim=0)
                        loss_test = self.loss_func(output, label)
                        acc_loss_test += loss_test.detach().cpu().item()
                        num_sequences += 1

                        # if 'lstm_attn_weights' in attns:
                        #     attns_weights.append(attns['lstm_attn_weights'].squeeze().detach().cpu().numpy().tolist())
                        # if 'pool_score' in attns:
                        #     node_attn = {}
                        #     node_attn["original_batch"] = sequence.batch.detach().cpu().numpy().tolist()
                        #     node_attn["pool_perm"] = attns['pool_perm'].detach().cpu().numpy().tolist()
                        #     node_attn["pool_batch"] = attns['batch'].detach().cpu().numpy().tolist()
                        #     node_attn["pool_score"] = attns['pool_score'].detach().cpu().numpy().tolist()
                        #     node_attns.append(node_attn)

                        # log metrics for risky and non-risky clips separately.
                        if not pred_all:
                            preds = torch.argmax(output)
                        else:
                            preds = output.max(1)[1].type_as(label)
                            
                        # ---------------------------------------- omg... ----------------------------------------
                        if(1 in label):
                            num_risky_sequences += 1
                            sum_seq_len += seq_len
                            if (1 in preds):
                                correct_risky_seq += 1 #sequence level metrics
                                if not pred_all:
                                    sum_prediction_frame = 0
                                    sum_predicted_risky_indices = 0
                                else:
                                    sum_prediction_frame += torch.where(preds == 1)[0][0].item() #returns the first index of a "risky" prediction in this sequence.
                                    sum_predicted_risky_indices += torch.sum(torch.where(preds==1)[0] + 1).item() / np.sum(range(seq_len + 1)) #(1*index)/seq_len added to sum.
                            else:
                                incorrect_risky_seq += 1
                                if not pred_all:
                                    sum_prediction_frame = 0
                                else:
                                    sum_prediction_frame += seq_len #if no risky predictions are made, then add the full sequence length to running avg.
                        elif(0 in label):
                            num_safe_sequences += 1
                            if (0 in preds):
                                correct_safe_seq += 1  #sequence level metrics
                                if not pred_all:
                                    sum_predicted_safe_indices = 0
                                else:
                                    sum_predicted_safe_indices += torch.sum(torch.where(preds==0)[0] + 1).item() / np.sum(range(seq_len + 1)) #(1*index)/seq_len added to sum.
                            else:
                                incorrect_safe_seq += 1 
                        # ---------------------------------------- omg... ----------------------------------------
                                
        avg_risky_prediction_frame = sum_prediction_frame / num_risky_sequences #avg of first indices in a sequence that a risky frame is first correctly predicted.
        avg_risky_seq_len = sum_seq_len / num_risky_sequences #sequence length for comparison with the prediction frame metric. 
        avg_predicted_risky_indices = sum_predicted_risky_indices / num_risky_sequences
        avg_predicted_safe_indices = sum_predicted_safe_indices / num_safe_sequences
        seq_tpr = correct_risky_seq / num_risky_sequences
        seq_fpr = incorrect_safe_seq / num_safe_sequences
        seq_tnr = correct_safe_seq / num_safe_sequences
        seq_fnr = incorrect_risky_seq / num_risky_sequences
        if prof != None:
            prof_result = prof.key_averages().table(sort_by="cuda_time_total")

        return outputs, \
                labels, \
                acc_loss_test / num_sequences, \
                attns_weights, \
                node_attns, \
                avg_risky_prediction_frame, \
                avg_risky_seq_len, \
                avg_predicted_risky_indices, \
                avg_predicted_safe_indices, \
                inference_time, \
                prof_result, \
                seq_tpr, \
                seq_fpr, \
                seq_tnr, \
                seq_fnr

    def evaluate(self, current_epoch=None):
        metrics = {}
        outputs_train, \
        labels_train, \
        acc_loss_train, \
        attns_train, \
        node_attns_train, \
        train_avg_prediction_frame, \
        train_avg_seq_len, \
        avg_predicted_risky_indices, \
        avg_predicted_safe_indices, \
        train_inference_time, \
        train_profiler_result, \
        seq_tpr, \
        seq_fpr, \
        seq_tnr, \
        seq_fnr = self.inference(self.training_data, self.training_labels, mode=self.config.inference_mode)

        metrics['train'] = get_metrics(outputs_train, labels_train)
        metrics['train']['loss'] = acc_loss_train
        metrics['train']['avg_prediction_frame'] = train_avg_prediction_frame
        metrics['train']['avg_seq_len'] = train_avg_seq_len
        metrics['train']['avg_predicted_risky_indices'] = avg_predicted_risky_indices
        metrics['train']['avg_predicted_safe_indices'] = avg_predicted_safe_indices
        metrics['train']['seq_tpr'] = seq_tpr
        metrics['train']['seq_tnr'] = seq_tnr
        metrics['train']['seq_fpr'] = seq_fpr
        metrics['train']['seq_fnr'] = seq_fnr
        with open("graph_profile_metrics.txt", mode='w') as f:
            f.write(train_profiler_result)

        outputs_test, \
        labels_test, \
        acc_loss_test, \
        attns_test, \
        node_attns_test, \
        val_avg_prediction_frame, \
        val_avg_seq_len, \
        avg_predicted_risky_indices, \
        avg_predicted_safe_indices, \
        test_inference_time, \
        test_profiler_result, \
        seq_tpr, \
        seq_fpr, \
        seq_tnr, \
        seq_fnr = self.inference(self.testing_data, self.testing_labels, mode=self.config.inference_mode)

        metrics['test'] = get_metrics(outputs_test, labels_test)
        metrics['test']['loss'] = acc_loss_test
        metrics['test']['avg_prediction_frame'] = val_avg_prediction_frame
        metrics['test']['avg_seq_len'] = val_avg_seq_len
        metrics['test']['avg_predicted_risky_indices'] = avg_predicted_risky_indices
        metrics['test']['avg_predicted_safe_indices'] = avg_predicted_safe_indices
        metrics['test']['seq_tpr'] = seq_tpr
        metrics['test']['seq_tnr'] = seq_tnr
        metrics['test']['seq_fpr'] = seq_fpr
        metrics['test']['seq_fnr'] = seq_fnr
        metrics['avg_inf_time'] = (train_inference_time + test_inference_time) / (len(labels_train) + len(labels_test))

        print("\ntrain loss: " + str(acc_loss_train) + ", acc:", metrics['train']['acc'], metrics['train']['confusion'], "mcc:", metrics['train']['mcc'], \
              "\ntest loss: " +  str(acc_loss_test) + ", acc:",  metrics['test']['acc'],  metrics['test']['confusion'], "mcc:", metrics['test']['mcc'])
        
        self.update_best_metrics(metrics, current_epoch)
        metrics['best_epoch'] = self.best_epoch
        metrics['best_val_loss'] = self.best_val_loss
        metrics['best_val_acc'] = self.best_val_acc
        metrics['best_val_auc'] = self.best_val_auc
        metrics['best_val_conf'] = self.best_val_confusion
        metrics['best_val_f1'] = self.best_val_f1
        metrics['best_val_mcc'] = self.best_val_mcc
        metrics['best_val_acc_balanced'] = self.best_val_acc_balanced
        metrics['best_avg_pred_frame'] = self.best_avg_pred_frame
        
        if self.config.n_folds <= 1 or self.log:
            log_wandb(metrics)
        
        return outputs_train, labels_train, outputs_test, labels_test, metrics


        #automatically save the model and metrics with the lowest validation loss
    def update_best_metrics(self, metrics, current_epoch):
        if metrics['test']['loss'] < self.best_val_loss:
            self.best_val_loss = metrics['test']['loss']
            self.best_epoch = current_epoch if current_epoch != None else self.config.epochs
            self.best_val_acc = metrics['test']['acc']
            self.best_val_auc = metrics['test']['auc']
            self.best_val_confusion = metrics['test']['confusion']
            self.best_val_f1 = metrics['test']['f1']
            self.best_val_mcc = metrics['test']['mcc']
            self.best_val_acc_balanced = metrics['test']['balanced_acc']
            self.best_avg_pred_frame = metrics['test']['avg_prediction_frame']
            #self.save_model()


    # Averages metrics after the end of each cross validation fold
    def update_cross_valid_metrics(self, outputs_train, labels_train, outputs_test, labels_test, metrics):
        if self.fold == 1:
            self.results['outputs_train'] = outputs_train
            self.results['labels_train'] = labels_train
            self.results['train'] = metrics['train']
            self.results['train']['loss'] = metrics['train']['loss']
            self.results['train']['avg_prediction_frame'] = metrics['train']['avg_prediction_frame'] 
            self.results['train']['avg_seq_len']  = metrics['train']['avg_seq_len'] 
            self.results['train']['avg_predicted_risky_indices'] = metrics['train']['avg_predicted_risky_indices'] 
            self.results['train']['avg_predicted_safe_indices'] = metrics['train']['avg_predicted_safe_indices']

            self.results['outputs_test'] = outputs_test
            self.results['labels_test'] = labels_test
            self.results['test'] = metrics['test']
            self.results['test']['loss'] = metrics['test']['loss'] 
            self.results['test']['avg_prediction_frame'] = metrics['test']['avg_prediction_frame'] 
            self.results['test']['avg_seq_len'] = metrics['test']['avg_seq_len'] 
            self.results['test']['avg_predicted_risky_indices'] = metrics['test']['avg_predicted_risky_indices'] 
            self.results['test']['avg_predicted_safe_indices'] = metrics['test']['avg_predicted_safe_indices']
            self.results['avg_inf_time'] = metrics['avg_inf_time']

            self.results['best_epoch']    = metrics['best_epoch']
            self.results['best_val_loss'] = metrics['best_val_loss']
            self.results['best_val_acc']  = metrics['best_val_acc']
            self.results['best_val_auc']  = metrics['best_val_auc']
            self.results['best_val_conf'] = metrics['best_val_conf']
            self.results['best_val_f1']   = metrics['best_val_f1']
            self.results['best_val_mcc']  = metrics['best_val_mcc']
            self.results['best_val_acc_balanced'] = metrics['best_val_acc_balanced']
            self.results['best_avg_pred_frame'] = metrics['best_avg_pred_frame']
        else:
            self.results['outputs_train'] = torch.cat((self.results['outputs_train'], outputs_train), dim=0)
            self.results['labels_train']  = torch.cat((self.results['labels_train'], labels_train), dim=0)
            self.results['train']['loss'] = np.append(self.results['train']['loss'], metrics['train']['loss'])
            self.results['train']['avg_prediction_frame'] = np.append(self.results['train']['avg_prediction_frame'], 
                                                                            metrics['train']['avg_prediction_frame'])
            self.results['train']['avg_seq_len']  = np.append(self.results['train']['avg_seq_len'], metrics['train']['avg_seq_len'])
            self.results['train']['avg_predicted_risky_indices'] = np.append(self.results['train']['avg_predicted_risky_indices'], 
                                                                                    metrics['train']['avg_predicted_risky_indices'])
            self.results['train']['avg_predicted_safe_indices'] = np.append(self.results['train']['avg_predicted_safe_indices'], 
                                                                                    metrics['train']['avg_predicted_safe_indices'])
            
            self.results['outputs_test'] = torch.cat((self.results['outputs_test'], outputs_test), dim=0)
            self.results['labels_test']  = torch.cat((self.results['labels_test'], labels_test), dim=0)
            self.results['test']['loss'] = np.append(self.results['test']['loss'], metrics['test']['loss'])
            self.results['test']['avg_prediction_frame'] = np.append(self.results['test']['avg_prediction_frame'], 
                                                                        metrics['test']['avg_prediction_frame'])
            self.results['test']['avg_seq_len'] = np.append(self.results['test']['avg_seq_len'], metrics['test']['avg_seq_len'])
            self.results['test']['avg_predicted_risky_indices'] = np.append(self.results['test']['avg_predicted_risky_indices'], 
                                                                                    metrics['test']['avg_predicted_risky_indices'])
            self.results['test']['avg_predicted_safe_indices'] = np.append(self.results['test']['avg_predicted_safe_indices'], 
                                                                                    metrics['test']['avg_predicted_safe_indices'])
            self.results['avg_inf_time'] = np.append(self.results['avg_inf_time'], metrics['avg_inf_time'])

            self.results['best_epoch']    = np.append(self.results['best_epoch'], metrics['best_epoch'])
            self.results['best_val_loss'] = np.append(self.results['best_val_loss'], metrics['best_val_loss'])
            self.results['best_val_acc']  = np.append(self.results['best_val_acc'], metrics['best_val_acc'])
            self.results['best_val_auc']  = np.append(self.results['best_val_auc'], metrics['best_val_auc'])
            self.results['best_val_conf'] = np.append(self.results['best_val_conf'], metrics['best_val_conf'])
            self.results['best_val_f1']   = np.append(self.results['best_val_f1'], metrics['best_val_f1'])
            self.results['best_val_mcc']  = np.append(self.results['best_val_mcc'], metrics['best_val_mcc'])
            self.results['best_val_acc_balanced'] = np.append(self.results['best_val_acc_balanced'], metrics['best_val_acc_balanced'])
            self.results['best_avg_pred_frame'] = np.append(self.results['best_avg_pred_frame'], metrics['best_avg_pred_frame'])
            
        # Log final averaged results
        if self.fold == self.config.n_folds:
            final_results = {}
            final_results['train'] = get_metrics(self.results['outputs_train'], self.results['labels_train'])
            final_results['train']['loss'] = np.average(self.results['train']['loss'])
            final_results['train']['avg_prediction_frame'] = np.average(self.results['train']['avg_prediction_frame'])
            final_results['train']['avg_seq_len'] = np.average(self.results['train']['avg_seq_len'])
            final_results['train']['avg_predicted_risky_indices'] = np.average(self.results['train']['avg_predicted_risky_indices'])
            final_results['train']['avg_predicted_safe_indices'] = np.average(self.results['train']['avg_predicted_safe_indices'])
            
            final_results['test'] = get_metrics(self.results['outputs_test'], self.results['labels_test'])
            final_results['test']['loss'] = np.average(self.results['test']['loss'])
            final_results['test']['avg_prediction_frame'] = np.average(self.results['test']['avg_prediction_frame'])
            final_results['test']['avg_seq_len'] = np.average(self.results['test']['avg_seq_len'])
            final_results['test']['avg_predicted_risky_indices'] = np.average(self.results['test']['avg_predicted_risky_indices'])
            final_results['test']['avg_predicted_safe_indices'] = np.average(self.results['test']['avg_predicted_safe_indices'])
            final_results['avg_inf_time'] = np.average(self.results['avg_inf_time'])

            # Best results
            final_results['best_epoch']    = np.average(self.results['best_epoch'])
            final_results['best_val_loss'] = np.average(self.results['best_val_loss'])
            final_results['best_val_acc']  = np.average(self.results['best_val_acc'])
            final_results['best_val_auc']  = np.average(self.results['best_val_auc'])
            final_results['best_val_conf'] = self.results['best_val_conf']
            final_results['best_val_f1']   = np.average(self.results['best_val_f1'])
            final_results['best_val_mcc']  = np.average(self.results['best_val_mcc'])
            final_results['best_val_acc_balanced'] = np.average(self.results['best_val_acc_balanced'])
            final_results['best_avg_pred_frame'] = np.average(self.results['best_avg_pred_frame'])

            print('\nFinal Averaged Results')
            print("\naverage train loss: " + str(final_results['train']['loss']) + ", average acc:", final_results['train']['acc'], final_results['train']['confusion'], final_results['train']['auc'], \
                "\naverage test loss: " +  str(final_results['test']['loss']) + ", average acc:", final_results['test']['acc'],  final_results['test']['confusion'], final_results['test']['auc'])

            log_wandb(final_results)
            
            return self.results['outputs_train'], self.results['labels_train'], self.results['outputs_test'], self.results['labels_test'], final_results

    def save_model(self):
        """Function to save the model."""
        saved_path = Path(self.config.model_save_path).resolve()
        os.makedirs(os.path.dirname(saved_path), exist_ok=True)
        torch.save(self.model.state_dict(), str(saved_path))
        with open(os.path.dirname(saved_path) + "/model_parameters.txt", "w+") as f:
            f.write(str(self.config))
            f.write('\n')
            f.write(str(' '.join(sys.argv)))

    def load_model(self):
        """Function to load the model."""
        saved_path = Path(self.config.model_load_path).resolve()
        if saved_path.exists():
            self.build_model()
            self.model.load_state_dict(torch.load(str(saved_path)))
            self.model.eval()
