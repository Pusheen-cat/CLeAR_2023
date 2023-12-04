import haiku as hk
from training import CFA_constants as constants
from training import curriculum as curriculum_lib
import jax
import jax.numpy as jnp
import numpy as np

def count_to7(x):
    arr = []
    for idx in range(8):
        t=jnp.where(x==idx, 1,0)
        arr.append(jnp.sum(t).item())
    return arr
'''
'modular_arithmetic', 'parity_check', 'even_pairs', 'cycle_navigation'

'divide_by_two': 'ndcf', ########
'reverse_string': 'dcf',
'stack_manipulation': 'dcf', ########
'compare_occurrence': 'dcf',

'binary_addition': 'cs',########
'binary_multiplication': 'cs',########
'bucket_sort': 'cs',
'compute_sqrt': 'cs',
'duplicate_string': 'cs',
'interlocked_pairing': 'cs',########
'odds_first': 'cs',
'''
task_name = 'odds_first' #'modular_arithmetic', 'parity_check', 'even_pairs', 'cycle_navigation'
task = constants.TASK_BUILDERS[task_name](fa_dim=4,)
see_position = False

sampling = 10_000
curriculum = curriculum_lib.UniformCurriculum(values=list(range(1, 40 + 1)))
pre_tune_rng_seq = hk.PRNGSequence(0)
pre_tune_rng_seq.reserve(sampling)  # random seed
out = np.array([0,0,0,0,0,0,0,0,])
for step in range(sampling):
    length = curriculum.sample_sequence_length(step)
    train_batch = task.sample_batch(next(pre_tune_rng_seq), length=length, batch_size=128)

    input = train_batch['input']
    batch_,l,dim = input.shape
    if see_position == False:
        input = input[:, :, -3:]
        for idx in range(batch_):
            for idxx in range(l):
                num = jnp.sum(input[idx, idxx]*jax.numpy.array([1,2,4]))
                out[int(num)] +=1
    else:
        input = input[:, :, 0]
        input = jnp.argmax(input, axis=-1)
        for idx in input:
            out[(idx*8)//l] +=1



    print(out/sum(out))

''' 
'modular_arithmetic' = [1.02 1.01 1.02 1.78 1.03 1.02 1.00 1.79]
'parity_check' = [1.01015942 1.02453892 1.02625821 1.01875586 1.         1.01187871 1.01891216 1.01203501]
'even_pairs' = [1.05188945 1.         1.01663847 1.02284264 1.01663847 1.01748449 1.02763677 1.00479413]
'cycle_navigation' = [1.52 1.01 1.00 1.01 1.52 1.01 1.02 1.01]
'''

