import numpy as np
from numpy import genfromtxt
import torch
import argparse
import yaml
from model import Ensemble
from utils import plot_one, plot_many, read_test_csv
import sys
import random
import copy
from sklearn.metrics import mean_squared_error

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(params: dict):

    np.random.seed(params['seed'])
    random.seed(params['seed'])

    orig_data = genfromtxt(params['data_dir'], delimiter=',', skip_header=1, usecols=(1,2,3,4,7))

    np_orig_data = np.array(orig_data)
    np_orig_data_mod = copy.deepcopy(np_orig_data)
    np_orig_data_mod[:,0] = np.mod(np_orig_data_mod[:,0], 2*np.pi)

    params['input_data_mod'] = np_orig_data_mod
    params['input_data'] = np_orig_data[:-1,:5]
    params['output_data'] = np_orig_data[1:,:4]
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

        noise_level = params['load_model_dir'].split('/')[-1].split('_')[-3]
        assert noise_level in params['data_dir'].split('/')[-1], "The model loaded is not trained on the loaded data..."
        
        ensemble_ins.load_model(params['load_model_dir']) # will reset val and train data 

        CIs = [0.67, 1.28, 1.96] # 50%, 80%, and 95% confidence intervals 
        all_mus = np.zeros((params['num_models'],4,1001))
        all_lower_mus = np.zeros((params['num_models'],4,1001))
        all_upper_mus = np.zeros((params['num_models'],4,1001))

        selected_mus = np.zeros((len(CIs),4,1001))
        selected_lower_mus = np.zeros((len(CIs),4,1001))
        selected_upper_mus = np.zeros((len(CIs),4,1001))

        ci_idx = 0
        for ci in CIs:
            print(f'Confidence Interval {ci}')
            for model_no, model in ensemble_ins.models.items():

                start = 0
                steps = 1001
                if params['pred_on_data'] == 'pred_on_train':
                    # for multi-step pred
                    first_input = ensemble_ins.rand_input_filtered_mod_train[start,:]
                    full_input = ensemble_ins.rand_input_train
                    # for one-step pred
                    all_inputs = ensemble_ins.rand_input_filtered_mod_train[start:start+steps,:]
                    orig_inputs = ensemble_ins.rand_input_train[start:start+steps,:4]
                    # common between both 
                    ground_truth = ensemble_ins.rand_output_train[start:start+steps,:]
                elif params['pred_on_data'] == 'pred_on_val':
                    """
                    # for multi-step pred
                    first_input = ensemble_ins.rand_input_filtered_mod_val[start,:]
                    full_input = ensemble_ins.rand_input_val 
                    # for one-step pred
                    all_inputs = ensemble_ins.rand_input_filtered_mod_val[start:start+steps,:]
                    orig_inputs = ensemble_ins.rand_input_val[start:start+steps,:4]                
                    # common between both
                    ground_truth = ensemble_ins.rand_output_val[start:start+steps,:]   
                    """
                    assert params['pred_on_data'].split('_')[-1] in params['test_data_dir'].split('/')[-1], "Provide the correct csv file..."
                    assert params['test_data_dir'] is not None, "You need to provide val data for validation..." 
                    assert noise_level in params['test_data_dir'].split('/')[-1], "The model loaded is not trained on the loaded data..."

                    filtered_input_data, input_data, output_data = read_test_csv(params['test_data_dir'], ensemble_ins.input_filter)
                    # for multi-step pred
                    first_input = filtered_input_data[start,:]
                    full_input = input_data
                    # for one-step pred
                    all_inputs = filtered_input_data[start:start+steps,:]
                    orig_inputs = input_data[start:start+steps,:4]
                    # common between both
                    ground_truth = output_data[start:start+steps,:]
                elif params['pred_on_data'] == 'pred_on_test':
                    assert params['pred_on_data'].split('_')[-1] in params['test_data_dir'].split('/')[-1], "Provide the correct csv file..."
                    assert params['test_data_dir'] is not None, "You need to provide test data for testing..." 
                    assert noise_level in params['test_data_dir'].split('/')[-1], "The model loaded is not trained on the loaded data..."

                    filtered_input_data, input_data, output_data = read_test_csv(params['test_data_dir'], ensemble_ins.input_filter)
                    # for multi-step pred
                    first_input = filtered_input_data[start,:]
                    full_input = input_data
                    # for one-step pred
                    all_inputs = filtered_input_data[start:start+steps,:]
                    orig_inputs = input_data[start:start+steps,:4]
                    # common between both
                    ground_truth = output_data[start:start+steps,:]

                pred_data = params['pred_on_data'].split('_')[-1] 

                if params['eval_mode'] == 'multistep':

                    mu, upper_mu, lower_mu = model.get_next_state_reward_free_sim(first_input, start,\
                                                                steps, ensemble_ins.input_filter_mod, full_input, ci=ci)
                    if model_no == 1 or model_no == 0:
                        plot_many(mu.T, upper_mu.T, lower_mu.T, ground_truth.T,\
                            no_of_outputs=4, save_dir="deep_ensemble/", file_name=f"model_{model_no}_pred_{start}_{start+steps}_multistep_{noise_level}_{pred_data}.png")
                elif params['eval_mode'] == 'single':
                    mu, logvar =  model.get_next_state_reward_one_step(all_inputs, \
                                                            deterministic=True, return_mean=False) # normalized validation data
                    mu_unnorm, upper_mu_unnorm, lower_mu_unnorm =  ensemble_ins.calculate_bounds(mu, logvar, orig_inputs, ci)
                    if model_no == 1 or model_no == 0:
                        plot_many(mu_unnorm.T, upper_mu_unnorm.T, lower_mu_unnorm.T, ground_truth.T,\
                            no_of_outputs=4, save_dir="deep_ensemble/", file_name=f"model_{model_no}_pred_{start}_{start+steps}_onestep_{noise_level}_{pred_data}.png")
                    mu = mu_unnorm
                    lower_mu = lower_mu_unnorm
                    upper_mu = upper_mu_unnorm
                
                all_mus[model_no,:,:] = mu.T
                all_lower_mus[model_no,:,:] = lower_mu.T
                all_upper_mus[model_no,:,:] = upper_mu.T

            # pick the best model according to MSE
            best_mu = np.argmin(np.array([mean_squared_error(ground_truth.T, all_mus[i,:,:]) for i in range(params['num_models'])]))
            selected_mus[ci_idx,:,:] = all_mus[best_mu,:,:] 
            selected_lower_mus[ci_idx,:,:] = all_lower_mus[best_mu,:,:] # theta, theta_d, x, x_d
            selected_upper_mus[ci_idx,:,:] = all_upper_mus[best_mu,:,:]
            ci_idx += 1

        selected_lower_mus[:,[1,2],:] = selected_lower_mus[:,[2,1],:] # theta, x, theta_d, x_d
        selected_upper_mus[:,[1,2],:] = selected_upper_mus[:,[2,1],:]
        selected_upper_mus = selected_upper_mus.reshape(len(CIs),-1)
        selected_lower_mus = selected_lower_mus.reshape(len(CIs),-1)
        # saving lower and upper confidence bounds in a csv file     
        stacked_CIs = np.stack((selected_lower_mus[0,:], selected_lower_mus[1,:], selected_lower_mus[2,:], selected_upper_mus[0,:],\
                                selected_upper_mus[1,:], selected_upper_mus[2,:]), axis=1)
        np.savetxt(f"./pred_csv/{noise_level}_{params['pred_on_data'].split('_')[-1]}_{params['eval_mode']}.csv", stacked_CIs, delimiter=',')


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
    parser.add_argument('--eval_mode', type=str, default=None, choices=('single', 'multistep'))
    parser.add_argument('--pred_on_data', type=str, choices=['pred_on_train', 'pred_on_val', 'pred_on_test'], default=None)
    parser.add_argument('--test_data_dir', type=str, default=None, help='path to csv file containing testing data')

    args = parser.parse_args()
    params = vars(args)

    if params['yaml_file']:
        with open(args.yaml_file, 'r') as f:
            yaml_config = yaml.load(f, Loader=yaml.FullLoader)
            for config in yaml_config['args']:
                if config in params:
                    params[config] = yaml_config['args'][config]

    train(params) 

    return                

if __name__ == '__main__':
    main()
