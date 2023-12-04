import haiku as hk
import numpy as np
from training import constants as constants
from training import curriculum as curriculum_lib
import pickle

class task_loader:
    def __init__(self, task_name, train_step, batch_size):
        self.batch_size = batch_size
        self.task = constants.TASK_BUILDERS[task_name]()
        self.curriculum = curriculum_lib.UniformCurriculum(values=list(range(1, 40 + 1)))
        self.pre_tune_rng_seq = hk.PRNGSequence(0)
        self.pre_tune_rng_seq.reserve(train_step)  # random seed
    def get_item(self, idx):
        length = self.curriculum.sample_sequence_length(idx)
        train_batch = self.task.sample_batch(next(self.pre_tune_rng_seq), length=length, batch_size=self.batch_size)
        return np.array(train_batch['input']), length

task_name = 'cycle_navigation' #'modular_arithmetic', 'parity_check', 'even_pairs', 'cycle_navigation'
train_step = 10_000
batch_size = 128

save = []
loader = task_loader(task_name, train_step, batch_size)
for step in range(train_step):
    data, length = loader.get_item(step)
    save.append([data,length])

with open(file='data.pickle', mode='wb') as f:
    pickle.dump(save, f)
