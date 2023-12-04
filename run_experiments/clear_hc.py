import copy
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


#static setting
univ_setting = 1 # This indicates it is basic scenario
complexity_parameter = [2,3,4,5,6,7,8] #Active if univ_setting ==1
cuda = 2
cl_setting = str([True, True, 1]) #{'cl':True, 'save pre_model make psudo label using it':True, setting:}
fa_dim = 4
pretune = str({'tune': True,'pretune_steps':1_000,'train_layers':['lin_in_','lin_out_'],})
lr_schedule = 0
share_input = 1
developing = 1

#dynamic
seed_list = [0]
tasks_list = [
    'modular_arithmetic_cl',
    'cycle_navigation_cl',
    'reverse_string_cl',
    'bucket_sort_cl',
]
joint_list = [0,1]
arch_list = ['rnn', 'lstm', 'stack_rnn', 'tape_rnn',]
step_list = [50_000]



task_idx = 0
for idx, task in enumerate(tasks_list):
    tasks = [task for i in range(len(complexity_parameter))]
    print(tasks)
    tasks = ' '.join(tasks)
    complexity_parameter_hc = ' '.join([str(x) for x in complexity_parameter])
    for joint in joint_list:
        for architecture in arch_list:
            for training_steps in step_list: #[]:
                for seed in seed_list:
                    task_idx += 1

                    exp_idx = copy.deepcopy(task_idx)
                    #if exp_idx in done: continue
                    #if not exp_idx in todo_: continue
                    print(f'#### Run {exp_idx}')
                    print(f"python all_argpars.py --univ_setting {univ_setting} --exp_idx {exp_idx} --cuda {cuda} "
                              f"--seed {seed} --tasks {tasks} --joint {joint} --architecture {architecture} "
                              f"--training_steps {training_steps} --fa_dim {fa_dim} --lr_schedule {lr_schedule} "
                              f"--share_input {share_input} --developing {developing} "
                              f"--complexity_parameter_hc {complexity_parameter_hc} ")
                    os.system(f"python all_argpars.py --univ_setting {univ_setting} --exp_idx {exp_idx} --cuda {cuda} "
                              f"--seed {seed} --tasks {tasks} --joint {joint} --architecture {architecture} "
                              f"--training_steps {training_steps} --fa_dim {fa_dim} --lr_schedule {lr_schedule} "
                              f"--share_input {share_input} --developing {developing} "
                              f"--complexity_parameter_hc {complexity_parameter_hc} ")

'''
parser.add_argument('--univ_setting', default=0,  type=int)
parser.add_argument('--exp_idx', default=-1,  type=int)
parser.add_argument('--cuda', default=0,  type=int)
parser.add_argument('--seed', default=0,  type=int)
parser.add_argument('--tasks', default=str(['even_pairs', 'parity_check', 'cycle_navigation', 'modular_arithmetic', ]),  type=str)
parser.add_argument('--joint', default=bool,  type=False)
parser.add_argument('--cl_setting', default=str((True, True, 1)),  type=str)
parser.add_argument('--architecture', default='lstm',  type=str)
parser.add_argument('--training_steps', default=50000,  type=int)
parser.add_argument('--fa_dim', default=3,  type=int)
parser.add_argument('--pretune', default=str({'tune': True, 'pretune_steps': 1_000, 'train_layers': ['lin_in_','lin_out_'],}),  type=str)
parser.add_argument('--lr_schedule', default=False,  type=bool)
parser.add_argument('--share_input', default=True,  type=bool)
parser.add_argument('--developing', default=True,  type=bool)
'''