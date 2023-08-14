import subprocess
import sys
import numpy as np

noises = ["det", "low_noise", "high_noise"]
names = ["train", "test", "val"] 
eval_modes = ["multistep", "single"]
dl_models = ["Model_seed0_2023_08_03_21-50-21_new_mod_det_nonoise_train",\
             "Model_seed0_2023_08_06_12-06-39_new_mod_low_noise_train",\
                "Model_seed0_2023_08_06_12-20-10_new_mod_high_noise_train"]

yaml_file = 'model_params.yml'

all_data = np.zeros((len(noises),len(names),len(eval_modes),2,3,4,1001)) #Example: low_noise,train,multistep,upper_mu,80%-CI,4-outputs,time-steps

for noise in noises:
    for name in names:
        for eval_mode in eval_modes:
            for dl_model in dl_models: 
                if noise in dl_model:
                    load_model_dir = f"./../../learnedModels/deep_ensemble/checkpoints/model_saved_weights/{dl_model}"
            data_dir = f"../../../Data/cartpoleData/{noise}_train.csv"
            pred_on_data = f"pred_on_{name}" 
            if name == 'test':
                test_data_dir = f"../../../Data/cartpoleData/{noise}_test.csv" 
            elif name == 'val':
                test_data_dir = f"../../../Data/cartpoleData/{noise}_val.csv"
            else:
                test_data_dir = "./"  # not important
            l = ["python",
                r"train.py",
                "--yaml_file",
                yaml_file,
                "--seed",
                str(0),
                "--data_dir",
                data_dir,
                "--load_model_dir",
                load_model_dir,
                "--eval_mode",
                eval_mode,
                "--pred_on_data",
                pred_on_data,
                "--test_data_dir",
                test_data_dir
                ]
            print("--data_dir",
                data_dir,
                "--eval_mode",
                eval_mode,
                "--pred_on_data",
                pred_on_data,
                )
            res = subprocess.run(l, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            for stdot in res.stdout.splitlines():
                print(stdot)
            for stder in res.stderr.splitlines():
                print(stder)     
            #print(f"Standard Error is: {res.stderr.splitlines()}")
            #sys.exit()
            #all_data[noise_idx,name_idx,eval_mode_idx,0,:,:,:], all_data[noise_idx,name_idx,eval_mode_idx,1,:,:,:] = res.stdout[0], res.stdout[1]
