# TrueLearn
The code that allows you to run truelearn experiments that tests the performance of TrueLearn: A bayesian learning 
strategy that models lifelong learner engagement incorperating background knowledge of the learner and novelty of educational resources.


# Citing the paper
This work is related to the paper `TrueLearn: A Family of Bayesian Algorithms to Match Lifelong Learners to Open Educational Resources` published at the Thirty-forth AAAI Conference on Artifical Intelligence, 2020 at New York, NY, USA. 

The bibtex entry for the publication is as follows:
```
@inproceedings{truelearn2020,
	author = {Bulathwela, S. and Perez-Ortiz, M. and Yilmaz, E. and Shawe-Taylor, J.},
	title={TrueLearn: A Family of Bayesian Algorithms to Match Lifelong Learners to Open Educational Resources},
	booktitle = {AAAI Conference on Artificial Intelligence},
	year = {2020}
}
```

About the files
===============

All the useful files in this module incorporate different algorithms used in the experiments

- `naive_baseline_models.py`: file contains the programming logic of persistence, majority models
- `multi_skill_kt_models.py`: file contains the programming logic of multi-skill knowledge tracing models
- `fixed_depth_truelearn_models.py`: file contains the programming logic of fixed-depth truelearn models
- `truelearn_models.py`: file contains the programming logic of TrueLearn Novel model

Additionally, there are two files that are used to run the experiment:
- `run_sequencial_trueskill_baseline.py`: file that is used to run Vanilla TrueSkill baseline and TrueLearn dynamic-depth as they are sequential models
- `run_experiments.py`: file that is used to run all the other experiments

Other helper files:
- `utils.py`: file contains required utility functions
- `analyse_results.py`: file used to produce precision, recall, accuracy and F1 metrics from the algorithms
