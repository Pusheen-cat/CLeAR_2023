# CLeAR: Continual Learning on Algorithmic Reasoning for Human-like Intelligence
### The code cleanup is still in progress.
<p align="center">
  <img width="50%" src="https://github.com/Pusheen-cat/CLeAR_2023/assets/50083459/47fa7de4-593a-4c69-b567-86c3312164d2" alt="Overview figure"/>
</p>

This repository provides an implementation of the paper [CLeAR: Continual Learning on Algorithmic Reasoning for Human-like Intelligence](https://openreview.net/forum?id=hz33V7Tb2O).

> Continual learning (CL) aims to incrementally learn multiple tasks that are presented sequentially. The significance of 
> CL lies not only in the practical importance but also in studying the learning mechanisms of humans who are excellent 
> continual learners. While most research on CL has been done on structured data such as images, there is a lack of 
> research on CL for abstract logical concepts such as counting, sorting, and arithmetic, which humans learn gradually 
> over time in the real world. In this work, for the first time, we introduce novel algorithmic reasoning (AR) methodology 
> for continual tasks of abstract concepts: CLeAR. Our methodology proposes a one-to-many mapping of input distribution 
> to a shared mapping space, which allows the alignment of various tasks of different dimensions and shared semantics. 
> Our tasks of abstract logical concepts, in the form of formal language, can be classified into Chomsky hierarchies 
> based on their difficulty. In this study, we conducted extensive experiments consisting of 15 tasks with various 
> levels of Chomsky hierarchy, ranging from in-hierarchy to inter-hierarchy scenarios. CLeAR not only achieved near zero 
> forgetting but also improved accuracy during following tasks, a phenomenon known as backward transfer, while previous 
> CL methods designed for image classification drastically failed.

It contains all code, datasets, and models necessary to reproduce the paper's results. 

This was built based on the [NNCH](https://github.com/google-deepmind/neural_networks_chomsky_hierarchy), so it shares many parts of the code.
## Content

```
.
├── make_encoder              - This is pytorch code
|   ├── balance_calc.py       - Calculated each task's mapping space noise
|   ├── encoder.py            - Training encoder in all-position independent sequence
|   └── loader.py             - Create dataset for encoder training
├── models
|   ├── ndstack_rnn.py        - Nondeterministic Stack-RNN (DuSell & Chiang, 2021)
|   ├── rnn.py                - RNN (Elman, 1990)
|   ├── stack_rnn.py          - Stack-RNN (Joulin & Mikolov, 2015)
|   ├── tape_rnn.py           - Tape-RNN, loosely based on Baby-NTM (Suzgun et al., 2019) 
|   ├── transformer.py        - Transformer (Vaswani et al., 2017)
|   └── others                - Modifications for continual learning (with mapping)
├── tasks
|   ├── cs                    - Context-sensitive tasks
|   ├── dcf                   - Determinisitc context-free tasks
|   ├── ndcf                  - Nondeterministic context-free tasks
|   ├── regular               - Regular tasks
|   └── task.py               - Abstract GeneralizationTask 
├── run_experiments           - Run each experiment
├── training
|   ├── ~constants.py         - Training/Evaluation constants
|   ├── curriculum.py         - Training curricula (over sequence lengths)
|   ├── ~example.py           - Example training script (RNN on the Even Pairs task)
|   ├── ~range_evaluation.py  - Evaluation loop (over unseen sequence lengths)
|   ├── ~training.py          - Training loop
|   └── ~utils.py             - Utility functions
├── visualization             - Visualization of model internal state
├── README.md
└── requirements.txt          - Dependencies
```
*Same as [NNCH](https://github.com/google-deepmind/neural_networks_chomsky_hierarchy)
'tasks' contains all tasks, organized in their Chomsky hierarchy levels (regular, dcf, cs). They all inherit the abstract class GeneralizationTask, defined in tasks/task.py.

'models' contains all the models we use, written in [jax](https://github.com/google/jax) and [haiku](https://github.com/deepmind/dm-haiku), two open source libraries.

'training' contains the code for training models and evaluating them on a wide
range of lengths. We also included an example to train and evaluate an RNN
on the Even Pairs task. We use [optax](https://github.com/deepmind/optax) for our optimizers.

## Installation

`pip install -r requirements.txt`


## Usage Example

`python run_experiments/clear_in.py`


## Citing This Work

```bibtex
@inproceedings{kang2023clear,
  title={CLeAR: Continual Learning on Algorithmic Reasoning for Human-like Intelligence},
  author={Kang, Bong Gyun and Kim, HyunGi and Jung, Dahuin and Yoon, Sungroh},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```


## License and Disclaimer

This code is based on the code from https://github.com/google-deepmind/neural_networks_chomsky_hierarchy. 
Attribution is required when used, and commercial use is prohibited.
