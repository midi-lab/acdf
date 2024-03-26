# Readme

This repository contains the code for the paper "Multistep Inverse Is Not All You Need," by Alexander Levine, Peter Stone, and Amy Zhang.

## Numerical Simulations

The ``numerical_experiments`` directory contains code for running the numerical experiments. These are enirely contained in one file, ``numerical_experiments.py``, which will run numerical simulations of AC-State and ACDF on tabular environments and return the success rates of these simulations. Results are both printed and saved to a .pth file. Lines commented out at the end of ``numerical_experiments.py`` control which of the four tabular environments in the paper to test, how for how many simulations of that environment to run, and how many transitions to collect; see comments in the file for a full description.

We ran these experiments with python 3.9.6, and the packages pytorch (2.2.1, on CPU) and more_itertools (10.2.0).

## Deep RL Experiments

The  ``gridworld_exploration`` directory contains code for the deep RL experiments in the paper. This code is forked from the code for the paper "Guaranteed Discovery of Control-Endogenous Latent States with Multi-Step Inverse Models," by Lamb et al. (2022), which is available at https://github.com/alexmlamb/ControllableLatentState/tree/main/gridworld_exploration. We provide an annotated diff demonstrating exactly how we modified their code for our experiments. The experiments in our paper can be reproduced using the following lines:


For the "baseline" maze environment, using AC-State:

```bash
python3 main.py --data maze  --k_steps ${K}  --ncodes ${NCODES}  --seed ${SEED}  --exo_noise two_maze --num_exo 8 --env_iteration 5000 --policy_selection random --no_reset_actions   --stochastic_start stochastic --ep_length 5000 --model_train_iter 5000 --eval_iter 5000 --num_iter 30000   --log_eval_prefix ./path/to/eval/log --no_restart true --use_best_model 
```

For the "baseline" maze environment, using ACDF (that is, using a forward model loss):

```bash
python3 main.py --data maze  --k_steps ${K}  --ncodes ${NCODES}  --seed ${SEED}  --exo_noise two_maze --num_exo 8 --env_iteration 5000 --policy_selection random --no_reset_actions   --stochastic_start stochastic --ep_length 5000 --model_train_iter 5000 --eval_iter 5000 --num_iter 30000   --log_eval_prefix ./path/to/eval/log --no_restart true --use_best_model --use_forward 
```

For the "periodic" environment, using AC-State:

```bash
python3 main.py --data periodic-cart  --k_steps ${K}  --ncodes ${NCODES}  --seed ${SEED}  --exo_noise two_maze  --num_exo 8 --env_iteration 5000 --policy_selection random --no_reset_actions --stochastic_start stochastic --ep_length 200 --model_train_iter 5000 --eval_iter 5000 --num_iter 30000  --log_eval_prefix ./path/to/eval/log --no_restart false --use_best_model 
```

For the "periodic" environment, using ACDF:

```bash
python3 main.py --data periodic-cart  --k_steps ${K}  --ncodes ${NCODES}  --seed ${SEED}  --exo_noise two_maze  --num_exo 8 --env_iteration 5000 --policy_selection random --no_reset_actions --stochastic_start stochastic --ep_length 200 --model_train_iter 5000 --eval_iter 5000 --num_iter 30000  --log_eval_prefix ./path/to/eval/log --no_restart false --use_best_model --use_forward
```

In addition to printing the success rate of open-loop planning trials, these commands will also save this information in files specified by the ``--log_eval_prefix`` option, with filenames in the format ``${PATH_PREFIX}_it_${ENV_ITERATION}_train_it_${TRAINING_ITERATION}_seed_${SEED}.pth``. This .pth file will contain a python dictionary where the "wins" key specifies the number of planning trials out of 1000 for which the learned encoder allowed for correct open-loop planning.

The ``environment.yaml`` file can be used for setting up a conda environment with the appropriate dependencies.



