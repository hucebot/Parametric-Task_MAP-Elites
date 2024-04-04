# Parametric-Task MAP-Elites 

This repository archives the Python scripts used in the following publication:
Anne, Timothée et Mouret, Jean-Baptiste (2024). «Parametric-Task MAP-Elites». Companion Proceedings of the Conference on Genetic and Evolutionary Computation, GECCO 2024, Companion Volume, Melbourne, Australia, July 14-18, 2024. https://doi.org/10.1145/3638529.3653993.

## Abstract
Optimizing a set of functions simultaneously by leveraging their similarity is called multi-task optimization. Current black-box multi-task algorithms only solve a finite set of tasks, even when the tasks originate from a continuous space. In this paper, we introduce Parametric-Task MAP-Elites (PT-ME), a new black-box algorithm for continuous multi-task optimization problems. This algorithm (1) solves a new task at each iteration, effectively covering the continuous space, and (2) exploits a new variation operator based on local linear regression. The resulting dataset of solutions makes it possible to create a function that maps any task parameter to its optimal solution. We show that PT-ME outperforms all baselines, including the deep reinforcement learning algorithm PPO on two parametric-task toy problems and a robotic problem in simulation. 

![Teaser_figure_Parametric-Task](https://github.com/hucebot/Parametric-Task_MAP-Elites/assets/72027302/d2ebbc74-05da-487f-8d76-05fab2056c0d)

(a) Quality-Diversity is the problem of finding high-performing solutions with diverse behaviors. (b) Multi-task optimization is the problem of finding the optimum solutions for a finite set of tasks, each often characterized by a task parameter (or descriptor). (c) In this paper, we propose to extend the multi-task optimization problem to a continuous parametrization, which we call Parametric-Task Optimization. The goal is to be able to return the optimal solution for any task parameter.

## Content

### To_Run_PT-ME

#### Files
 - PT-ME (not cleaned).ipynb: one file executable (closest to the archived files)
 - PT-ME (cleaned).ipynb: one file executable with the minimal instructions

#### Requirements

 - pip install numpy
 - pip install scikit-learn
 - pip install tqdm
 - pip install matplotlib

   ### Archive

This folder only has an archive function and is not meant to be easily run.

#### Files

 - 03_cma-es.py: Execute the CMA-ES baselines on the two toy problems

 - 03_fast_random.py: Execute the Random baseline on the two toy problems

 - 09_ppo.py: Execute PPO on the three problems

 - 11_PT_MAP-Elites_figures.py: Main file for Parametric-Task MAP-Elites + comparisons and figures

 - 28_batch_NN_learning.py: Execute the distillation (28_learning_NN.py) in batch

 - 28_learning_NN.py: Distillate the dataset into a neural network

 - 28_PT_ME_dataset_for_NN.py: Reformate the evaluations for the distillation (28_learning_NN.py)

 - 28_PT_ME_generalization_figures.py: Inference evaluation

 - base_talos_01.py: Miscalenous functions (Countain the three problems definitions)

 - plot_01.py: For creating the figures.
