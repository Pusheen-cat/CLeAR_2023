import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from training.Old_er_tasks_main import main

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
dict_ = {}
dict_['univ_setting'] = 1  # This indicates it is basic scenario // Not used for old scenario yet
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


memory_size_list = [300]
task_name = 'er'
dim_match_method_list = ['padding', 'embedding', 'mapping']
seeds = [0]
tasks_list = [['modular_arithmetic_cl', 'modular_arithmetic_cl', 'modular_arithmetic_cl', 'modular_arithmetic_cl'], ]
joint_list = [False]
architecture = ['rnn', 'stack_rnn', 'tape_rnn', 'lstm', ]
step_list = [50_000]

dict_['fa_dim'] = 4
dict_['lr_schedule'] = False
dict_['developing'] = True
dict_['joint_training'] = False
dict_['pretune'] = {'tune': True, 'pretune_steps': 1_000, 'train_layers': ['lin_in_', 'lin_out_'], }

idx_ = 0
for memory_size in memory_size_list:
    cl_setting = (True, task_name, memory_size)

    for idx, tasks in enumerate(tasks_list):
        dict_['tasks'] = tasks

        for method in dim_match_method_list:
            dim_match_method = method

            for joint in joint_list:
                if joint:
                    dict_['cl_setting'] = (False, False, 1)
                else:
                    dict_['cl_setting'] = cl_setting

                for arch in architecture:
                    for seed in seeds:
                        dict_['seed'] = seed

                        if len(tasks) > 4:
                            mul_factor = 2
                        else:
                            mul_factor = 1
                        if arch == 'tape_rnn':
                            dict_['architecture_params'] = {'hidden_size': 256 * mul_factor,
                                                            'memory_cell_size': 8 * mul_factor, 'memory_size': 40}
                        elif arch == 'stack_rnn':
                            dict_['architecture_params'] = {'hidden_size': 256 * mul_factor,
                                                            'stack_cell_size': 8 * mul_factor, }
                        elif arch == 'lstm':
                            dict_['architecture_params'] = {'hidden_size': 256 * mul_factor, }
                        elif arch == 'rnn':
                            dict_['architecture_params'] = {'hidden_size': 256 * mul_factor, }
                        dict_['architecture'] = arch

                        for step in step_list:  # []:
                            dict_['training_steps'] = step

                            if dim_match_method == 'mapping':  # ours
                                dict_['use_raw'] = False
                                dict_['simple_share'] = False
                            elif dim_match_method == 'embedding':
                                dict_['use_raw'] = True
                                dict_['simple_share'] = False
                                dict_['projection_dim'] = 10
                            elif dim_match_method == 'padding':
                                dict_['use_raw'] = True
                                dict_['simple_share'] = True

                            idx_ += 1
                            print(f'#### Run {idx_} th')
                            dict_['exp_idx'] = idx_
                            main(dict_.copy())
