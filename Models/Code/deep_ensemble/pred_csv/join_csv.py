import numpy as np 
from numpy import genfromtxt
import pandas as pd
import sys, os

noises = ["det", "low", "high"]
names = ["train", "val", "test"] 
eval_modes = ["single", "multi"]

np.set_printoptions(suppress=True,formatter={'float_kind':'{:0.2f}'.format})
an_array_train = np.linspace(0,10.00,num=1001) 
time_steps_train = np.hstack((an_array_train, an_array_train, an_array_train, an_array_train))

an_array_val = np.linspace(500,510,num=1001) 
time_steps_val = np.hstack((an_array_val, an_array_val, an_array_val, an_array_val))

dfs_variable, dfs_t, dfs_lower_50, dfs_lower_80, dfs_lower_95, dfs_upper_50, dfs_upper_80, dfs_upper_95,\
      dfs_name, dfs_eval_mode, dfs_noise = [], [], [], [], [], [], [], [], [], [], []

for noise in noises:
    for name in names:
        for eval_mode in eval_modes:
            file_name = f'./{noise}_{name}_{eval_mode}step.csv' if eval_mode == "multi" else f'./{noise}_{name}_{eval_mode}.csv'
            df = pd.read_csv(file_name, header=None,low_memory=False, encoding='UTF-8')
            df = df.rename(columns={0:"lower_50",1:"lower_80",2:"lower_95",3:"upper_50",4:"upper_80",5:"upper_95"})
            if name == 'train' or name == 'test':
                df["t"] = time_steps_train
            elif name == 'val':
                df["t"] = time_steps_val    
            df["name"] = [name for i in range(4004)]
            df["eval_mode"] = [eval_mode for i in range(4004)]
            noise_name = f"{noise}_noise" if noise == "low" or noise == "high" else f"{noise}"
            df["noise"] = [noise_name for i in range(4004)]
            df["variable"] = ['theta' for i in range(1001)] + ['x' for i in range(1001)] + ['theta_d' for i in range(1001)] + ['x_d' for i in range(1001)]
            dfs_variable.append(df['variable'])
            dfs_t.append(df["t"])
            dfs_lower_50.append(df["lower_50"])
            dfs_lower_80.append(df["lower_80"])
            dfs_lower_95.append(df["lower_95"])
            dfs_upper_50.append(df["upper_50"])
            dfs_upper_80.append(df["upper_80"])
            dfs_upper_95.append(df["upper_95"])
            dfs_name.append(df["name"])
            dfs_eval_mode.append(df["eval_mode"])
            dfs_noise.append(df["noise"])
            #df = pd.DataFrame(df)
            #print(df)

df_t = pd.concat([df for df in dfs_t], ignore_index=True)
df_variable = pd.concat([df for df in dfs_variable], ignore_index=True)
df_lower_50 = pd.concat([df for df in dfs_lower_50], ignore_index=True)
df_lower_80 = pd.concat([df for df in dfs_lower_80], ignore_index=True)
df_lower_95 = pd.concat([df for df in dfs_lower_95], ignore_index=True)
df_upper_50 = pd.concat([df for df in dfs_upper_50], ignore_index=True)
df_upper_80 = pd.concat([df for df in dfs_upper_80], ignore_index=True)
df_upper_95 = pd.concat([df for df in dfs_upper_95], ignore_index=True)
df_name = pd.concat([df for df in dfs_name], ignore_index=True)
df_eval_mode = pd.concat([df for df in dfs_eval_mode], ignore_index=True)
df_noise = pd.concat([df for df in dfs_noise], ignore_index=True)


all_cols = {"variable": df_variable, "t": df_t, "lower_50": df_lower_50, "lower_80": df_lower_80,\
            "lower_95": df_lower_95, "upper_50": df_upper_50, "upper_80": df_upper_80, "upper_95": df_upper_95,\
            "name": df_name, "eval_mode": df_eval_mode, "noise": df_noise}
pd.DataFrame(all_cols).to_csv("deepensemble_pred.csv", index=False)
#df_f.to_csv("deepensemble_pred.csv") 
