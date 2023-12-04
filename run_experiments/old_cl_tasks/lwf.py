import copy
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

#static setting
univ_setting = 0 # This indicates it is basic scenario
cuda = 0
old_cl_method = 'lwf'#'lwf'
old_cl_method_parms = 0 #0 for default lwf

fa_dim = 4
lr_schedule = 0
share_input = 1
developing = 1
joint_list = [0]

#dynamic
seed_list = [0]
tasks_list = [
    ['even_pairs', 'cycle_navigation', 'parity_check', 'compare_occurrence', 'reverse_string',
     'stack_manipulation', 'interlocked_pairing', 'odds_first', 'bucket_sort', 'duplicate_string'],
]
dim_match_method_list = ['mapping', 'embedding', 'padding']
arch_list = ['rnn', 'lstm',]
step_list = [50_000]


start = 1
end = 1000
task_idx = 0
for idx, tasks in enumerate(tasks_list):
    tasks = ' '.join(tasks)
    for joint in joint_list:
        for dim_match_method in dim_match_method_list:
            for architecture in arch_list:
                for training_steps in step_list: #[]:
                    for seed in seed_list:
                        task_idx += 1
                        if task_idx <start:
                            continue
                        else:
                            exp_idx = copy.deepcopy(task_idx)
                        if task_idx >= end:
                            raise 'End training'

                        print(f'#### Run {exp_idx}')
                        print(f"")
                        os.system(f"python old_parser.py --univ_setting {univ_setting} --exp_idx {exp_idx} --cuda {cuda} "
                                  f"--seed {seed} --tasks {tasks} --joint {joint} --architecture {architecture} "
                                  f"--training_steps {training_steps} --fa_dim {fa_dim} --lr_schedule {lr_schedule} "
                                  f"--share_input {share_input} --developing {developing} --dim_match_method {dim_match_method} "
                                  f"--old_cl_method {old_cl_method} --old_cl_method_parms {old_cl_method_parms} --sheet_name {sheet_name} ")