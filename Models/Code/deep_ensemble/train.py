import numpy as np
from numpy import genfromtxt
import torch
import argparse
import yaml
from model import Ensemble
from utils import plot_one, plot_many
import sys
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(params: dict):

    np.random.seed(params['seed'])
    random.seed(params['seed'])

    orig_data = genfromtxt(params['data_dir'], delimiter=',')
    params['input_data'] = np.array(orig_data[:-1,:5])
    params['output_data'] = np.array(orig_data[1:,:4])
    params['delta'] = params['output_data'] - params['input_data'][:,:4]

    params['no_of_inputs'] = params['input_data'].shape[1]
    params['no_of_outputs'] = params['output_data'].shape[1]

    ensemble_ins = Ensemble(params=params)
    # calculate mean and variance of input/output data
    ensemble_ins.calculate_mean_var() 
    ensemble_ins.set_loaders() # load the saved data during testing 
    if params['train_mode']:
        ensemble_ins.train_model(params['model_epochs'], True, params['min_model_epochs'])
    if params['test_mode']:
        ensemble_ins.load_model(params['load_model_dir']) # will reset val and train data 
        for model_no, model in ensemble_ins.models.items():
            if params['free_sim_mode']:
                ground_truth = ensemble_ins.rand_output_val[-900:,:]
                mu, upper_mu, lower_mu = model.get_next_state_reward_free_sim(ensemble_ins.rand_input_filtered_val[0,:],\
                                                            900, ensemble_ins.input_filter, ensemble_ins.rand_input_filtered_val)
                plot_many(mu.T, upper_mu.T, lower_mu.T, ground_truth.T,\
                       no_of_outputs=4, save_dir="deep_ensemble/", file_name=f"model_{model_no}_pred.png")
            else:
                ground_truth = ensemble_ins.rand_output_val[:2400,:]
                mu, logvar =  model.get_next_state_reward_one_step(ensemble_ins.rand_input_filtered_val[:2400,:], \
                                                        deterministic=True, return_mean=False) # normalized validation data
                mu_unnorm, upper_mu_unnorm, lower_mu_unnorm =  ensemble_ins.calculate_bounds(mu, logvar)
                plot_many(mu_unnorm.T, upper_mu_unnorm.T, lower_mu_unnorm.T, ground_truth.T,\
                       no_of_outputs=4, save_dir="deep_ensemble/", file_name=f"model_{model_no}_pred.png")

    return 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-se', type=int, default=0)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--yaml_file', '-yml', type=str, default=None)
    parser.add_argument('--num_models', '-nm', type=int, default=7)
    parser.add_argument('--train_val_ratio', type=float, default=0.2)
    parser.add_argument('--model_epochs', '-me', type=int, default=200)
    parser.add_argument('--model_lr', type=float, default=0.001, help='lr for Transition Model')
    parser.add_argument('--l2_reg_multiplier', type=float, default=1.)
    parser.add_argument('--min_model_epochs', type=int, default=None)
    parser.add_argument('--train_mode', type=bool, default=False)
    parser.add_argument('--test_mode', type=bool, default=False)
    parser.add_argument('--load_model_dir', type=str, default=None)
    parser.add_argument('--free_sim_mode', type=str, default=None)

    args = parser.parse_args()
    params = vars(args)

    if params['yaml_file']:
        with open(args.yaml_file, 'r') as f:
            yaml_config = yaml.load(f, Loader=yaml.FullLoader)
            for config in yaml_config['args']:
                if config in params:
                    params[config] = yaml_config['args'][config]

    train(params)                

if __name__ == '__main__':
    main()
