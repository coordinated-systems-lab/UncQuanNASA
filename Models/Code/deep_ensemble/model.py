import itertools
import random
import math
from collections import deque, namedtuple
from typing import List
import time
import sys
from copy import deepcopy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler
import math
import tabulate
import datetime
from utils import GaussianMSELoss, MeanStdevFilter, prepare_data, check_or_make_folder
import pickle, copy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TransitionDataset(Dataset):
    ## Dataset wrapper for sampled transitions
    def __init__(self, input_data, delta):

        self.data_X = torch.Tensor(input_data)   
        self.data_y = torch.Tensor(delta)          

    def __len__(self):
        return len(self.data_X)

    def __getitem__(self, index):
        return self.data_X[index], self.data_y[index]


class EnsembleTransitionDataset(Dataset):
    ## Dataset wrapper for sampled transitions
    def __init__(self, input_data, delta, n_models=1):

        data_count = input_data.shape[0]

        idxs = np.random.randint(data_count, size=[n_models, data_count])
        self._n_models = n_models
        self.data_X = torch.Tensor(input_data[idxs])
        self.data_y = torch.Tensor(delta[idxs])
 
    def __len__(self):
        return self.data_X.shape[1]

    def __getitem__(self, index):
        return self.data_X[:, index], self.data_y[:, index]

class Ensemble(object):
    def __init__(self, params):

        self.params = params
        self.input_data_mod = params['input_data_mod']
        self.input_data = params['input_data']
        self.output_data = params['output_data']
        self.delta = params['delta']
        self.input_dim = params['no_of_inputs']
        self.output_dim = params['no_of_outputs']
        self.models = {i: Model(input_dim=self.input_dim,
                                output_dim=self.output_dim,
                                seed=params['seed'] + i,
                                l2_reg_multiplier=params['l2_reg_multiplier'],
                                num=i)
                       for i in range(params['num_models'])}
        #print(f"FIrst loss: {self.models[0].model.state_dict()['logvar.bias']}")
        self.num_models = params['num_models']
        self.train_val_ratio = params['train_val_ratio']
        self._model_lr = params['model_lr'] if 'model_lr' in params else 0.001
        weights = [weight for model in self.models.values() for weight in model.weights]
        # initializing the max and min logvar to bound the predicted variance  
        self.max_logvar = torch.full((self.output_dim,), 0.5, requires_grad=True, device=device)
        self.min_logvar = torch.full((self.output_dim,), -10.0, requires_grad=True, device=device)
        weights.append({'params': [self.max_logvar]}) #  learning the max and min logvar
        weights.append({'params': [self.min_logvar]}) 
        self.set_model_logvar_limits()  

        self.optimizer = torch.optim.Adam(weights, lr=self._model_lr)
        self._lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.3, verbose=False)
        self.input_filter_mod = MeanStdevFilter(self.input_dim) 
        self.input_filter = MeanStdevFilter(self.input_dim) 
        #self.output_filter = MeanStdevFilter(self.output_dim)

        self._model_id = "Model_seed{}_{}_{}".format(params['seed'],\
                                                datetime.datetime.now().strftime('%Y_%m_%d_%H-%M-%S'), 'new_mod_high_noise_train')

    def calculate_mean_var(self) -> None:

        total_points = self.input_data.shape[0]

        for i in range(total_points):
            self.input_filter_mod.update(self.input_data_mod[i,:])
            self.input_filter.update(self.input_data[i,:])

        return 
    
    def set_loaders(self):

        input_filtered = prepare_data(self.input_data, self.input_filter)
        input_filtered_mod = prepare_data(self.input_data_mod, self.input_filter_mod)
        #output_filtered = prepare_data(self.output_data, self.output_filter)
        self.train_size = int((1-self.train_val_ratio)*input_filtered.shape[0])
        self.val_size = input_filtered.shape[0] - self.train_size

        ########## MIX VALIDATION AND TRAINING ##########

        #randperm = np.random.permutation(self.train_size + self.val_size)
        randperm = np.linspace(0, stop=(input_filtered.shape[0]-1), num=input_filtered.shape[0], dtype=int)
        randperm_train, randperm_val = randperm[:self.train_size], randperm[self.train_size:]

        self.rand_input_train_mod = self.input_data_mod[randperm_train,:]
        self.rand_input_val_mod = self.input_data_mod[randperm_val,:]

        self.rand_input_train = self.input_data[randperm_train,:]
        self.rand_input_val = self.input_data[randperm_val,:]

        self.rand_output_train = self.output_data[randperm_train,:]
        self.rand_output_val = self.output_data[randperm_val,:]

        self.rand_input_filtered_mod_train = input_filtered_mod[randperm_train,:]
        self.rand_input_filtered_mod_val = input_filtered_mod[randperm_val,:]

        self.rand_input_filtered_train = input_filtered[randperm_train,:]
        self.rand_input_filtered_val = input_filtered[randperm_val,:]

        self.rand_delta_filtered_train = self.delta[randperm_train,:]
        self.rand_delta_filtered_val = self.delta[randperm_val,:]

        batch_size = 256

        self.transition_loader = DataLoader(
            EnsembleTransitionDataset(self.rand_input_filtered_mod_train, self.rand_delta_filtered_train, n_models=self.num_models),
            shuffle=True,
            batch_size=batch_size,
            pin_memory=True
        )
        
        validate_dataset = TransitionDataset(self.rand_input_filtered_mod_val[-1000:], self.rand_delta_filtered_val[-1000:])
        sampler = SequentialSampler(validate_dataset)
        self.validation_loader = DataLoader(
            validate_dataset,
            sampler=sampler,
            batch_size=batch_size,
            pin_memory=True
        )

    def train_model(self, max_epochs: int = 100, save_model=False, min_model_epochs=None):
        self.current_best_losses = np.zeros(          # params['num_models'] = 7
            self.params['num_models']) + sys.maxsize  # weird hack (YLTSI), there's almost surely a better way...
        self.current_best_weights = [None] * self.params['num_models']
        val_improve = deque(maxlen=157) #6 originally
        lr_lower = False
        min_model_epochs = 0 if not min_model_epochs else min_model_epochs

        ### check validation before first training epoch
        improved_any, iter_best_loss = self.check_validation_losses(self.validation_loader)
        val_improve.append(improved_any)
        best_epoch = 0
        model_idx = 0
        print('Epoch: %s, Total Loss: N/A' % (0))
        print('Validation Losses:')
        print('\t'.join('M{}: {}'.format(i, loss) for i, loss in enumerate(iter_best_loss)))
        for i in range(max_epochs):  # 1000
            t0 = time.time()
            total_loss = 0
            loss = 0
            step = 0
            # value to shuffle dataloader rows by so each epoch each model sees different data
            perm = np.random.choice(self.num_models, size=self.num_models, replace=False)
            for x_batch, diff_batch in self.transition_loader:  

                x_batch = x_batch[:, perm]
                diff_batch = diff_batch[:, perm]
                step += 1
                for idx in range(self.num_models):
                    loss += self.models[idx].train_model_forward(x_batch[:, idx], diff_batch[:, idx])  
                total_loss = loss.item()
                loss += 0.01 * self.max_logvar.sum() - 0.01 * self.min_logvar.sum()
                self.optimizer.zero_grad()
                loss.backward()  
                self.optimizer.step()
                loss = 0
  
            t1 = time.time()
            print("Epoch training took {} seconds".format(t1 - t0))
            if (i + 1) % 1 == 0:
                improved_any, iter_best_loss = self.check_validation_losses(self.validation_loader)
                print('Epoch: {}, Total Loss: {}'.format(int(i + 1), float(total_loss)))
                print('Validation Losses:')
                print('\t'.join('M{}: {}'.format(i, loss) for i, loss in enumerate(iter_best_loss)))
                print('Best Validation Losses So Far:')
                print('\t'.join('M{}: {}'.format(i, loss) for i, loss in enumerate(self.current_best_losses)))
                val_improve.append(improved_any)
                if improved_any:
                    best_epoch = (i + 1)
                    print('Improvement detected this epoch.')
                else:
                    epoch_diff = i + 1 - best_epoch
                    plural = 's' if epoch_diff > 1 else ''
                    print('No improvement detected this epoch: {} Epoch{} since last improvement.'.format(epoch_diff,plural))
                if len(val_improve) > 156:
                    if not any(np.array(val_improve)[1:]):  # If no improvement in the last 5 epochs
                        # assert val_improve[0]
                        if (i >= min_model_epochs):
                            print('Validation loss stopped improving at %s epochs' % (best_epoch))
                            for model_index in self.models:
                                self.models[model_index].load_state_dict(self.current_best_weights[model_index])
                            #self._select_elites(validation_loader)
                            if save_model:
                                self._save_model()
                            return
                        elif not lr_lower:
                            self._lr_scheduler.step()
                            lr_lower = True
                            val_improve = deque(maxlen=6)
                            val_improve.append(True)
                            print("Lowering Adam Learning for fine-tuning")
                t2 = time.time() 
                print("Validation took {} seconds".format(t2 - t1))
        #self._select_elites(validation_loader)

    def _save_model(self):
        """
        Method to save model after training is completed
        """
        print("Saving model checkpoint...")
        check_or_make_folder("./../../learnedModels")
        check_or_make_folder("./../../learnedModels/deep_ensemble")
        check_or_make_folder("./../../learnedModels/deep_ensemble/checkpoints")
        check_or_make_folder("./../../learnedModels/deep_ensemble/checkpoints/model_saved_weights")
        save_dir = "./../../learnedModels/deep_ensemble/checkpoints/model_saved_weights/{}".format(self._model_id)
        check_or_make_folder(save_dir)
        # Create a dictionary with pytorch objects we need to save, starting with models
        torch_state_dict = {'model_{}_state_dict'.format(i): w for i, w in enumerate(self.current_best_weights)}
        # Then add logvariance limit terms
        torch_state_dict['logvar_min'] = self.min_logvar
        torch_state_dict['logvar_max'] = self.max_logvar
        # Save Torch files
        torch.save(torch_state_dict, save_dir + "/torch_model_weights.pt")
        # saving train and val data
        print("Saving train and val data...")           
        data_state_dict = {'train_input_data': self.rand_input_filtered_train, 
                           'val_input_data': self.rand_input_filtered_val,
                           'train_input_data_mod': self.rand_input_filtered_mod_train,
                           'val_input_data_mod': self.rand_input_filtered_mod_val,
                           'train_out_data': self.rand_delta_filtered_train,
                           'val_out_data': self.rand_delta_filtered_val,
                           'input_filter': self.input_filter,
                           'input_filter_mod': self.input_filter_mod}   
        pickle.dump(data_state_dict, open(save_dir + '/model_data.pkl', 'wb'))

    def check_validation_losses(self, validation_loader):
        improved_any = False
        current_losses, current_weights = self._get_validation_losses(validation_loader, get_weights=True)
        #print(f"Iter Best Loss: {current_losses}")
        #print(f"Current Best Losses: {self.current_best_losses}")
        improvements = ((self.current_best_losses - current_losses) / self.current_best_losses) > 0.01
        #print(f"Improvements :{improvements}")
        for i, improved in enumerate(improvements):
            if improved:
                self.current_best_losses[i] = current_losses[i]
                self.current_best_weights[i] = current_weights[i]
                improved_any = True
        return improved_any, current_losses

    def _get_validation_losses(self, validation_loader, get_weights=True):
        best_losses = []
        best_weights = []
        for model in self.models.values():
            best_losses.append(model.get_validation_loss(validation_loader))
            if get_weights:
                best_weights.append(deepcopy(model.state_dict()))
        best_losses = np.array(best_losses)
        return best_losses, best_weights   

    def set_model_logvar_limits(self):

        for model in self.models.values():
            model.model.update_logvar_limits(self.max_logvar, self.min_logvar) 

    def load_model(self, model_dir):
        """loads the trained models"""
        torch_state_dict = torch.load(model_dir + '/torch_model_weights.pt', map_location=device)
        for i in range(self.num_models):
            self.models[i].load_state_dict(torch_state_dict['model_{}_state_dict'.format(i)])
        self.min_logvar = torch_state_dict['logvar_min']
        self.max_logvar = torch_state_dict['logvar_max']
        # loading train and val data     
        data_state_dict = pickle.load(open(model_dir + '/model_data.pkl', 'rb'))
        self.rand_input_filtered_train = data_state_dict['train_input_data']
        self.rand_input_filtered_val = data_state_dict['val_input_data']
        self.rand_input_filtered_mod_train = data_state_dict['train_input_data_mod']
        self.rand_input_filtered_mod_val = data_state_dict['val_input_data_mod']
        self.rand_delta_filtered_train = data_state_dict['train_out_data']
        self.rand_delta_filtered_val = data_state_dict['val_out_data'] 
        self.input_filter = data_state_dict['input_filter']
        self.input_filter_mod = data_state_dict['input_filter_mod']
        # reinitialize the train and val loaders 
        batch_size = 256

        self.transition_loader = DataLoader(
            EnsembleTransitionDataset(self.rand_input_filtered_mod_train, self.rand_delta_filtered_train, n_models=self.num_models),
            shuffle=True,
            batch_size=batch_size,
            pin_memory=True
        )
        
        validate_dataset = TransitionDataset(self.rand_input_filtered_mod_val, self.rand_delta_filtered_val)
        sampler = SequentialSampler(validate_dataset)
        self.validation_loader = DataLoader(
            validate_dataset,
            sampler=sampler,
            batch_size=batch_size,
            pin_memory=True
        )

    def calculate_bounds(self, mu:torch.Tensor, logvar:torch.Tensor, last_time_step: np.ndarray, ci: float):
        """
        mu: unnormalized predictions in tensor
        logvar: predicted logvar in tensor
        """
        if len(mu.shape) == 1:
            mu = mu.reshape(1,-1)
            logvar = logvar.reshape(1,-1)

        upper_mu =  mu + torch.mul(logvar.exp().sqrt(), ci)
        lower_mu =  mu - torch.mul(logvar.exp().sqrt(), ci)

        mu = mu.detach().cpu().numpy()
        upper_mu = upper_mu.detach().cpu().numpy()
        lower_mu = lower_mu.detach().cpu().numpy()

        mu = mu + last_time_step
        upper_mu = upper_mu + last_time_step
        lower_mu = lower_mu + last_time_step

        return mu, upper_mu, lower_mu 

class Model(nn.Module):
    def __init__(self, input_dim: int,
                 output_dim: int,
                 h: int = 1024,
                 seed=0,
                 l2_reg_multiplier=1.,
                 num=0):

        super(Model, self).__init__()
        torch.manual_seed(seed)
        self.model = BayesianNeuralNetwork(input_dim, output_dim, 200, l2_reg_multiplier, seed)
        self.weights = self.model.weights

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def get_next_state_reward_one_step(self, input: torch.Tensor, deterministic=False, return_mean=False):
        return self.model.get_next_state_reward_one_step(input, deterministic, return_mean) 

    def get_next_state_reward_free_sim(self, input: torch.Tensor, start :int, steps:int, input_filter:MeanStdevFilter,\
                                        full_input:torch.Tensor, ci: float, deterministic=True, return_mean=False):
        """
        input: Initial state to start free simulation 
        steps: No of time steps to simulation aka rollout horizon    
        input_filter: To feed standardize data at next time-step
        full_input: To extract control inputs for rollout horizon    
        deterministic: True should be preferable 
        return_mean: False if deterministic is True  
        """
        return self.model.get_next_state_reward_free_sim(input, start, steps, input_filter, full_input, ci, deterministic, return_mean) 
    
    def _train_model_forward(self, x_batch):
        self.model.train()    # TRAINING MODE
        self.model.zero_grad()
        x_batch = x_batch.to(device, non_blocking=True)
        y_pred = self.forward(x_batch)
        return y_pred

    def train_model_forward(self, x_batch, delta_batch):
        delta_batch = delta_batch.to(device, non_blocking=True)
        y_pred = self._train_model_forward(x_batch)
        y_batch = delta_batch
        loss = self.model.loss(y_pred, y_batch)
        return loss

    def get_predictions_from_loader(self, data_loader, return_targets = False, return_sample=False):
        self.model.eval()   # EVALUATION MODE
        preds, targets = [], []
        with torch.no_grad():
            for x_batch_val, delta_batch_val in data_loader:
                x_batch_val, delta_batch_val = x_batch_val.to(device, non_blocking=True),\
                                                delta_batch_val.to(device, non_blocking=True)
                y_pred_val = self.forward(x_batch_val)
                preds.append(y_pred_val)
                if return_targets:
                    y_batch_val = delta_batch_val
                    targets.append(y_batch_val)

        preds = torch.vstack(preds)

        if return_sample:
            mu, logvar = preds.chunk(2, dim=1)
            dist = torch.distributions.Normal(mu, logvar.exp().sqrt())
            sample = dist.sample()
            preds = torch.cat((sample, preds), dim=1)

        if return_targets:
            targets = torch.vstack(targets)
            return preds, targets
        else:
            return preds
                
    def get_validation_loss(self, validation_loader):
        self.model.eval()
        preds, targets = self.get_predictions_from_loader(validation_loader, return_targets=True)

        return self.model.loss(preds, targets, logvar_loss=False).item()


class BayesianNeuralNetwork(nn.Module):
    def __init__(self, input_dim: int,
                 output_dim: int,
                 h: int = 200,
                 l2_reg_multiplier=1.,
                 seed=0):
        super().__init__()
        torch.manual_seed(seed)
        #del self.network
        self.fc1 = nn.Linear(input_dim, h)
        reinitialize_fc_layer_(self.fc1)
        self.fc2 = nn.Linear(h, h)
        reinitialize_fc_layer_(self.fc2)
        self.fc3 = nn.Linear(h, h)
        reinitialize_fc_layer_(self.fc3)
        self.fc4 = nn.Linear(h, h)
        reinitialize_fc_layer_(self.fc4)
        self.use_blr = False
        self.delta = nn.Linear(h, output_dim)
        reinitialize_fc_layer_(self.delta)
        self.logvar = nn.Linear(h, output_dim)
        reinitialize_fc_layer_(self.logvar)
        self.loss = GaussianMSELoss()
        self.activation = nn.SiLU()
        self.lambda_prec = 1.0
        self.max_logvar = None
        self.min_logvar = None
        params = [] # 12 dicts for all layers (w and b) from get_weight_bias_parameters_with_decays method 
        self.layers = [self.fc1, self.fc2, self.fc3, self.fc4, self.delta, self.logvar]
        self.decays = np.array([0.000025, 0.00005, 0.000075, 0.000075, 0.0001, 0.0001]) * l2_reg_multiplier
        for layer, decay in zip(self.layers, self.decays):
            params.extend(get_weight_bias_parameters_with_decays(layer, decay))
        self.weights = params
        self.to(device)

    @staticmethod
    def filter_inputs(input, input_filter):
        input_f = input_filter.filter_torch(input)
        return input_f   

    def get_l2_reg_loss(self):
        l2_loss = 0
        for layer, decay in zip(self.layers, self.decays):
            for name, parameter in layer.named_parameters():
                if 'weight' in name:
                    l2_loss += parameter.pow(2).sum() / 2 * decay
        return l2_loss

    def update_logvar_limits(self, max_logvar, min_logvar):
        self.max_logvar, self.min_logvar = max_logvar, min_logvar

    def forward(self, x: torch.Tensor):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        delta = self.delta(x)
        logvar = self.logvar(x)
        # Taken from the PETS code to stabilise training (https://github.com/kchua/handful-of-trials)
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        return torch.cat((delta, logvar), dim=1)

    def get_next_state_reward_one_step(self, input: torch.Tensor, deterministic=False, return_mean=False):
        input_torch = torch.FloatTensor(input).to(device)
        mu, logvar = self.forward(input_torch).chunk(2, dim=1)
        mu_orig = mu

        if not deterministic:
            dist = torch.distributions.Normal(mu, logvar.exp().sqrt())
            mu = dist.sample()
 
        if return_mean:
            mu = torch.cat((mu, mu_orig), dim=1)

        return mu, logvar
    
    def get_next_state_reward_free_sim(self, input:torch.Tensor, start :int, steps:int, input_filter:MeanStdevFilter,\
                                        full_input:torch.Tensor, ci: float, deterministic=True, return_mean=False): # aka multi-step transition 
        input = input.reshape(1,-1)
        mu = np.zeros((steps, input.shape[1]-1)) # we are not predicting control input so -1
        upper_mu = np.zeros((steps, input.shape[1]-1))
        lower_mu = np.zeros((steps, input.shape[1]-1))
        mu_orig = torch.zeros(steps, input.shape[1]-1, device=device)
        input_torch = torch.FloatTensor(input).to(device)
        full_input_torch = torch.FloatTensor(full_input).to(device)

        for step in range(steps):

            delta, logvar = self.forward(input_torch).chunk(2, dim=1)
            #input_unnorm = input_filter.invert_torch(input_torch) # shape: [1,5]
            delta_orig = delta

            if not deterministic:
                dist = torch.distributions.Normal(delta, logvar.exp().sqrt())
                delta = dist.sample()

            # return unnormalized upper and lower bounds along with mu
            if step == 0: 
                mu[step,:], upper_mu[step,:], lower_mu[step,:] = self.calculate_bounds(step, delta, logvar, full_input_torch[start,:4].reshape(1,-1), ci=ci)
            else:     
                mu[step,:], upper_mu[step,:], lower_mu[step,:] = self.calculate_bounds(step, delta, logvar, mu[step-1,:].reshape(1,-1), ci=ci)

            #if return_mean:
            #    mu_orig[step,:] = delta_orig + full_input_torch[start+step,:4].reshape(1,-1)

            if step == 0:
                pred_without_mod = delta + full_input_torch[start,:4].reshape(1,-1)
            else:
                pred_without_mod = delta + torch.FloatTensor(mu[step-1,:].reshape(1,-1)).to(device)  

            pred_without_mod[:,0] = torch.remainder(pred_without_mod[:,0], 2*np.pi)
            next_state_norm = input_filter.filter_torch(pred_without_mod) # filter for data with mod transformation

            input_torch = torch.cat((next_state_norm.reshape(1,-1), full_input_torch[start+step+1,4].reshape(1,-1)), dim=1)

        #if return_mean:
        #    mu_orig = mu_orig.detach().cpu().numpy()
        #    return (mu, mu_orig), upper_mu, lower_mu    

        return mu, upper_mu, lower_mu
    
    def calculate_bounds(self, step: int, delta:torch.Tensor, logvar:torch.Tensor, input_unnorm:torch.Tensor, ci: float):
        """
        mu: unnormalized delta predictions in tensor
        logvar: predicted logvar in tensor
        """
        if len(delta.shape) == 1:
            delta = delta.reshape(1,-1)
            logvar = logvar.reshape(1,-1)

        upper_delta =  delta + torch.mul(logvar.exp().sqrt(), ci)
        lower_delta =  delta - torch.mul(logvar.exp().sqrt(), ci)

        delta = delta.detach().cpu().numpy()
        upper_delta = upper_delta.detach().cpu().numpy()
        lower_delta = lower_delta.detach().cpu().numpy()
        if step == 0:
            input_unnorm = input_unnorm.detach().cpu().numpy()

        mu = delta + input_unnorm
        upper_mu = upper_delta + input_unnorm
        lower_mu = lower_delta + input_unnorm

        return mu, upper_mu, lower_mu 

def reinitialize_fc_layer_(fc_layer):
    """
    Helper function to initialize a fc layer to have a truncated normal over the weights, and zero over the biases
    """
    input_dim = fc_layer.weight.shape[1]
    std = get_trunc_normal_std(input_dim)
    torch.nn.init.trunc_normal_(fc_layer.weight, std=std, a=-2 * std, b=2 * std)
    torch.nn.init.zeros_(fc_layer.bias)


def get_trunc_normal_std(input_dim):
    """
    Returns the truncated normal standard deviation required for weight initialization
    """
    return 1 / (2 * np.sqrt(input_dim))

def get_weight_bias_parameters_with_decays(fc_layer, decay):
    """
    For the fc_layer, extract only the weight from the .parameters() method so we don't regularize the bias terms
    """
    decay_params = []
    non_decay_params = []
    for name, parameter in fc_layer.named_parameters():
        if 'weight' in name:
            decay_params.append(parameter)
        elif 'bias' in name:
            non_decay_params.append(parameter)

    decay_dicts = [{'params': decay_params, 'weight_decay': decay}, {'params': non_decay_params, 'weight_decay': 0.}]

    return decay_dicts