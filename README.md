# Responsible AI (RAI) Games and Ensembles 

## Quick Start

Before running the code, please install all the required packages in `requirements.txt` by running:
```shell
pip install -r requirements.txt
```

In the code, we solve linear programs with the MOSEK solver, which requires a license. You can acquire a free academic license from https://www.mosek.com/products/academic-licenses/. Please make sure that the license file is placed in the correct folder so that the solver could work.

### Illustration

To get acquainted with a simple variant of our algorithm, refer to `illustration.ipynb`.

### Description of Files

For running experiments, review the training scripts names as train_${dataset}.sh. Use the `--download` option to download the dataset if you are running for the first time. 

### Evaluation

To evaluate the models trained with the above command, run:
```
python eval_cvar_worst_loss.py --file <result>.mat
```

## Introduction

Several recent works have studied the societal effects of AI; these include issues such as fairness, robustness, and safety.  In these problems, a learner seeks to minimize its worst-case loss over a set of predefined distributions. In this work, we provide a general framework for studying these problems, which we refer to as Responsible AI (RAI) games. We provide two classes of algorithms for solving these games:  (a) game-play based algorithms, and (b) greedy stagewise estimation algorithms. The former class of algorithms is motivated by online learning and game theory, whereas the latter class is motivated by the classical statistical literature on boosting, and regression. Empirically we demonstrate the generality and superiority of our techniques for solving several RAI problems around subpopulation shift.

## Algorithms

| Name      | Description |
| ----------- | ----------- |
| uniform      |  All sample weight vectors are uniform distributions. |
| adaboost   | Adaboost  |
| raigame (--greedy)  | Greedy Variant of RAI game |
| raigame (--gameplay)  | Gameplay Variant of RAI game (FW Update)|
| raigame (--gameplay --gen_adaboost)  | Gameplay Variant of RAI game (Gen Adaboost Update) |

## Parameters
All default training parameters can be found in `config.py`. All of the parameters can also be set through the training scripts.