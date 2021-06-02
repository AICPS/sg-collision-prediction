import os, sys, pdb
sys.path.append(os.path.dirname(sys.path[0]))
from argparse import ArgumentParser
from .dpm_model import DeepPredictiveModel
from pathlib import Path
import torch
import torch.optim as optim
from torch_geometric.data import DataListLoader
from torch.utils.data import DataLoader, TensorDataset
import wandb
import pandas as pd
import numpy as np
import pickle as pkl
from tqdm import tqdm
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sg_risk_assessment.metrics import *

INPUT_SHAPE = (1, 1, 5, 1, 64, 64) # (num_camera, batch_size, frames, channels, height, width)

#model configuration settings. specified on the command line
class Config:
    def __init__(self, args):
        self.parser = ArgumentParser()
        self.parser.add_argument('--cache_path', type=str, default="../scripts/dpm_271_seqlen_5.pkl", help="Path to the cache file.")
        self.parser.add_argument('--transfer_path', type=str, default="", help="Path to the transfer file.")
        self.parser.add_argument('--model_load_path', type=str, default="./model/model_best_val_loss_.vec.pt", help="Path to load cached model file.")
        self.parser.add_argument('--model_save_path', type=str, default="./model/model_best_val_loss_.vec.pt", help="Path to save model file.")
        self.parser.add_argument('--n_folds', type=int, default=1, help='Number of folds for cross validation')
        self.parser.add_argument('--split_ratio', type=float, default=0.3, help="Ratio of dataset withheld for testing.")
        self.parser.add_argument('--downsample', type=lambda x: (str(x).lower() == 'true'), default=False, help='Set to true to downsample dataset.')
        self.parser.add_argument('--learning_rate', default=0.00005, type=float, help='The initial learning rate.')
        self.parser.add_argument('--seed', type=int, default=0, help='Random seed.')
        self.parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
        self.parser.add_argument('--activation', type=str, default='relu', help='Activation function to use, options: [relu, leaky_relu].')
        self.parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
        self.parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
        self.parser.add_argument('--batch_size', type=int, default=8, help='Number of sequences in a batch.')
        self.parser.add_argument('--device', type=str, default="cuda", help='The device on which models are run, options: [cuda, cpu].')
        self.parser.add_argument('--test_step', type=int, default=5, help='Number of training epochs before testing the model.')

        parsed_args = self.parser.parse_args(args)
        wandb.init(project="av-dpm")
        wandb_config = wandb.config

        for arg_name in vars(parsed_args):
            self.__dict__[arg_name] = getattr(parsed_args, arg_name)
            wandb_config[arg_name] = getattr(parsed_args, arg_name)

        self.cache_path = Path(self.cache_path).resolve()
        
        if os.path.exists(self.transfer_path) and os.path.splitext(self.transfer_path)[-1] == '.pkl':
            self.transfer_path = Path(self.transfer_path).resolve()
        else:
            self.transfer_path = None
            print('Not using transfer learning')


#This class trains and evaluates the DPM model
class DPMTrainer:
    def __init__(self, args):
        self.config = Config(args)
        self.args = args
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        self.best_val_loss = 99999
        self.best_epoch = 0
        self.best_val_acc = 0
        self.best_val_auc = 0
        self.best_val_confusion = []
        self.best_val_f1 = 0
        self.best_val_mcc = -1.0
        self.best_val_acc_balanced = 0
        self.log = False

        if not self.config.cache_path.exists():
            raise Exception("The cache file does not exist.")
        
        with open(self.config.cache_path, 'rb') as f:
            self.dataset = pkl.load(f)
            pdb.set_trace()

        if self.config.transfer_path != None and self.config.transfer_path.exists():
            with open(self.config.transfer_path, 'rb') as f:
                self.transfer = pkl.load(f)
        
        # Class balancer
        if self.config.downsample == True:
            self.dataset = self.balance_dataset(self.dataset)
            # Transfer balancer
            # if self.config.transfer_path != None:
                # self.transfer = self.balance_dataset(self.transfer)
        
        self.toGPU = lambda x, dtype: torch.as_tensor(x, dtype=dtype, device=self.config.device)
        self.split_dataset()
        self.build_model()


    # TODO: Ensure dataset has a diverse representation of risk and non risk lane changes
    # assumes label is the same for all frames in a scene
    def balance_dataset(self, dataset):
        # binary classes
        seq_label = lambda x: x[1][0]
        risk = [seq_label(sequence) for sequence in dataset if seq_label(sequence) == 1].count(1)
        non_risk = len(dataset) - risk
        min_number = min(risk, non_risk)
        risk = min_number
        non_risk = min_number

        balanced = []
        for sequence in dataset:
            label = seq_label(sequence)
            if label == 1 and risk > 0:
                risk -= 1
                balanced.append(sequence)
            if label == 0 and non_risk > 0:
                non_risk -= 1
                balanced.append(sequence)
            if risk == 0 and non_risk == 0:
                break

        return balanced

    def split_dataset(self):
        training_data, testing_data = train_test_split(self.dataset, test_size=self.config.split_ratio, shuffle=True, random_state=self.config.seed, stratify=None)
        self.dataset = None #clearing to save memory
        # transfer learning
        if self.config.transfer_path != None: 
            training_data = np.append(training_data, testing_data, axis=0)
            testing_data = self.transfer

        self.training_x, self.training_y, self.training_filenames = list(zip(*training_data))
        del training_data
        
        self.testing_x, self.testing_y, self.testing_filenames = list(zip(*testing_data))
        del testing_data
        
        if self.config.n_folds <= 1:
            print("Number of Training Sequences Included: ", len(self.training_x))
            print("Number of Testing Sequences Included: ", len(self.testing_x))

        self.training_filenames = np.concatenate([np.full(y.shape[0], int(fname)) for y,fname in zip(self.training_y, self.training_filenames)])
        self.testing_filenames  = np.concatenate([np.full(y.shape[0], int(fname)) for y,fname in zip(self.testing_y, self.testing_filenames)])
        self.training_x = np.concatenate(self.training_x)
        self.testing_x  = np.concatenate(self.testing_x)
        self.training_x = np.expand_dims(self.training_x, axis=-3) # color channels = 1
        self.testing_x  = np.expand_dims(self.testing_x, axis=-3) # color channels = 1
        self.training_y = np.concatenate(self.training_y)
        self.testing_y  = np.concatenate(self.testing_y)
        self.class_weights = torch.from_numpy(compute_class_weight('balanced', np.unique(self.training_y), self.training_y))
        
        if self.config.n_folds <= 1:
            print("Num of Training Labels in Each Class: " + str(np.unique(self.training_y, return_counts=True)[1]) + ", Class Weights: " + str(self.class_weights))
            print("Num of Testing Labels in Each Class: " + str(np.unique(self.testing_y, return_counts=True)[1]) + ", Class Weights: " + str(self.class_weights))


    def build_model(self):
        self.model = DeepPredictiveModel(INPUT_SHAPE, self.config).to(self.config.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        if self.class_weights.shape[0] < 2:
            self.loss_func = torch.nn.CrossEntropyLoss()
        else:    
            self.loss_func = torch.nn.CrossEntropyLoss(weight=self.class_weights.float().to(self.config.device))
        wandb.watch(self.model, log="all")


    # Pick between Standard Training and KFold Cross Validation Training
    def learn(self):
        if self.config.n_folds <= 1 or self.config.transfer_path != None:
            print('Running Standard Training Loop\n')
            self.train()
        else:
            print(torch.cuda.get_device_name(0))
            print('Running {}-Fold Cross Validation Training Loop\n'.format(self.config.n_folds))
            self.cross_valid()


    def cross_valid(self):

        # KFold cross validation with similar class distribution in each fold
        skf = StratifiedKFold(n_splits=self.config.n_folds)
        X = np.append(self.training_x, self.testing_x, axis=0)
        y = np.append(self.training_y, self.testing_y, axis=0)
        filenames = np.append(self.training_filenames, self.testing_filenames, axis=0)

        # self.results stores average metrics for the the n_folds
        self.results = {}
        self.fold = 1

        # Split training and testing data based on n_splits (Folds)
        for train_index, test_index in skf.split(X, y):
            self.training_x, self.testing_x, self.training_y, self.testing_y = None, None, None, None #clear vars to save memory
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            training_filenames, testing_filenames = filenames[train_index], filenames[test_index]
            self.class_weights = torch.from_numpy(compute_class_weight('balanced', np.unique(y_train), y_train))

            # Update dataset
            self.training_x = X_train
            self.testing_x  = X_test
            self.training_y = y_train
            self.testing_y  = y_test
            self.training_filenames = training_filenames
            self.testing_filenames  = testing_filenames
            
            print('\nFold {}'.format(self.fold))
            print("Number of Training Sequences Included: ", len(np.unique(training_filenames)))
            print("Number of Testing Sequences Included: ",  len(np.unique(testing_filenames)))
            print("Num of Training Labels in Each Class: " + str(np.unique(self.training_y, return_counts=True)[1]) + ", Class Weights: " + str(self.class_weights))
            print("Num of Testing Labels in Each Class: "  + str(np.unique(self.testing_y, return_counts=True)[1]) + ", Class Weights: " + str(self.class_weights))
            
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
            permutation = np.random.permutation(len(self.training_x)) # shuffle dataset before each epoch
            self.model.train()

            for i in range(0, len(self.training_x), self.config.batch_size): # iterate through batches of the dataset
                batch_index = i + self.config.batch_size if i + self.config.batch_size <= len(self.training_x) else len(self.training_x)
                indices = permutation[i:batch_index]
                batch_x, batch_y = self.training_x[indices], self.training_y[indices]
                batch_x, batch_y = self.toGPU(batch_x, torch.float32), self.toGPU(batch_y, torch.long)
                batch_x = torch.unsqueeze(batch_x, 0) #cameras = 1
                output = self.model.forward(batch_x).view(-1, 2)
                loss_train = self.loss_func(output, batch_y)
                loss_train.backward()
                acc_loss_train += loss_train.detach().cpu().item() * len(indices)
                self.optimizer.step()
                del loss_train

            acc_loss_train /= len(self.training_x)
            tqdm_bar.set_description('Epoch: {:04d}, loss_train: {:.4f}'.format(epoch_idx, acc_loss_train))
            
            # no cross validation 
            if epoch_idx % self.config.test_step == 0:
                self.evaluate(epoch_idx)
        
    
    def inference(self, testing_x, testing_y, testing_filenames):
        labels = torch.LongTensor().to(self.config.device)
        outputs = torch.FloatTensor().to(self.config.device)
        testing_filenames = torch.as_tensor(testing_filenames)
        acc_loss_test = 0
        sum_prediction_frame = 0
        sum_seq_len = 0
        num_risky_sequences = 0
        num_safe_sequences = 0
        sum_predicted_risky_indices = 0 #sum is calculated as (value * (index+1))/sum(range(seq_len)) for each value and index in the sequence.
        sum_predicted_safe_indices = 0  #sum is calculated as ((1-value) * (index+1))/sum(range(seq_len)) for each value and index in the sequence.
        inference_time = 0
        prof_result = ""
        batch_size = self.config.batch_size #NOTE: set to 1 when profiling or calculating inference time.
        correct_risky_seq = 0
        correct_safe_seq = 0
        incorrect_risky_seq = 0
        incorrect_safe_seq = 0

        with torch.autograd.profiler.profile(enabled=False, use_cuda=True) as prof:
            with torch.no_grad():
                self.model.eval()

                for i in range(0, len(testing_x), batch_size): # iterate through subsequences
                    batch_index = i + batch_size if i + batch_size <= len(testing_x) else len(testing_x)
                    batch_x, batch_y = testing_x[i:batch_index], testing_y[i:batch_index]
                    batch_x, batch_y = self.toGPU(batch_x, torch.float32), self.toGPU(batch_y, torch.long)
                    batch_x = torch.unsqueeze(batch_x, 0) #cameras = 1
                    #start = torch.cuda.Event(enable_timing=True)
                    #end =  torch.cuda.Event(enable_timing=True)
                    #start.record()
                    output = self.model.forward(batch_x).view(-1, 2)
                    #end.record()
                    #torch.cuda.synchronize()
                    inference_time += 0#start.elapsed_time(end)
                    loss_test = self.loss_func(output, batch_y)
                    acc_loss_test += loss_test.detach().cpu().item() * len(batch_y)
                    outputs = torch.cat([outputs, output], dim=0)
                    labels = torch.cat([labels,batch_y], dim=0)

        #extract list of sequences and their associated predictions. calculate metrics over sequences.
        sequences = torch.unique(testing_filenames)
        for seq in sequences:
            indices = torch.where(testing_filenames == seq)[0]
            seq_outputs = outputs[indices]
            seq_labels = labels[indices]

            #log metrics for risky and non-risky clips separately.
            if(1 in seq_labels):
                preds = seq_outputs.max(1)[1].type_as(seq_labels)
                num_risky_sequences += 1
                sum_seq_len += seq_outputs.shape[0]
                if (1 in preds):
                    correct_risky_seq += 1 #sequence level metrics
                    sum_prediction_frame += torch.where(preds == 1)[0][0].item() #returns the first index of a "risky" prediction in this sequence.
                    sum_predicted_risky_indices += torch.sum(torch.where(preds==1)[0]+1).item()/np.sum(range(seq_outputs.shape[0]+1))
                else:
                    incorrect_risky_seq += 1
                    sum_prediction_frame += seq_outputs.shape[0] #if no risky predictions are made, then add the full sequence length to running avg.
            elif(0 in seq_labels):
                preds = seq_outputs.max(1)[1].type_as(seq_labels)
                num_safe_sequences += 1
                if(1 in preds):
                    incorrect_safe_seq += 1
                else:
                    correct_safe_seq += 1 

                if (0 in preds):
                    sum_predicted_safe_indices += torch.sum(torch.where(preds==0)[0]+1).item()/np.sum(range(seq_outputs.shape[0]+1))

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
                acc_loss_test/len(testing_x), \
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
        train_avg_prediction_frame, \
        train_avg_seq_len, \
        avg_predicted_risky_indices, \
        avg_predicted_safe_indices, \
        train_inference_time, \
        train_profiler_result, \
        seq_tpr, seq_fpr, seq_tnr, seq_fnr = self.inference(self.training_x, 
                                                            self.training_y, 
                                                            self.training_filenames)
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
        with open("dpm_profile_metrics.txt", mode='w') as f:
            f.write(train_profiler_result)

        outputs_test, \
        labels_test, \
        acc_loss_test, \
        val_avg_prediction_frame, \
        val_avg_seq_len, \
        avg_predicted_risky_indices, \
        avg_predicted_safe_indices, \
        test_inference_time, \
        test_profiler_result, \
        seq_tpr, seq_fpr, seq_tnr, seq_fnr = self.inference(self.testing_x, 
                                                            self.testing_y, 
                                                            self.testing_filenames)
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
        metrics['avg_inf_time'] = (train_inference_time + test_inference_time) / ((len(self.training_y) + len(self.testing_y))*5)

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
            self.results['train']['avg_prediction_frame'] = np.append(self.results['train']['avg_prediction_frame'], metrics['train']['avg_prediction_frame'])
            self.results['train']['avg_seq_len']  = np.append(self.results['train']['avg_seq_len'], metrics['train']['avg_seq_len'])
            self.results['train']['avg_predicted_risky_indices'] = np.append(self.results['train']['avg_predicted_risky_indices'], metrics['train']['avg_predicted_risky_indices'])
            self.results['train']['avg_predicted_safe_indices'] = np.append(self.results['train']['avg_predicted_safe_indices'], metrics['train']['avg_predicted_safe_indices'])
            
            self.results['outputs_test'] = torch.cat((self.results['outputs_test'], outputs_test), dim=0)
            self.results['labels_test']  = torch.cat((self.results['labels_test'], labels_test), dim=0)
            self.results['test']['loss'] = np.append(self.results['test']['loss'], metrics['test']['loss'])
            self.results['test']['avg_prediction_frame'] = np.append(self.results['test']['avg_prediction_frame'], metrics['test']['avg_prediction_frame'])
            self.results['test']['avg_seq_len'] = np.append(self.results['test']['avg_seq_len'], metrics['test']['avg_seq_len'])
            self.results['test']['avg_predicted_risky_indices'] = np.append(self.results['test']['avg_predicted_risky_indices'], metrics['test']['avg_predicted_risky_indices'])
            self.results['test']['avg_predicted_safe_indices'] = np.append(self.results['test']['avg_predicted_safe_indices'], metrics['test']['avg_predicted_safe_indices'])
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


    #UNTESTED
    def save_model(self):
        """Function to save the model."""
        saved_path = Path(self.config.model_save_path).resolve()
        os.makedirs(os.path.dirname(saved_path), exist_ok=True)
        torch.save(self.model.state_dict(), str(saved_path))
        with open(os.path.dirname(saved_path) + "/model_parameters.txt", "w+") as f:
            f.write(str(self.config))
            f.write('\n')
            f.write(str(' '.join(sys.argv)))

    #UNTESTED
    def load_model(self):
        """Function to load the model."""
        saved_path = Path(self.config.model_load_path).resolve()
        if saved_path.exists():
            self.build_model()
            self.model.load_state_dict(torch.load(str(saved_path)))
            self.model.eval()