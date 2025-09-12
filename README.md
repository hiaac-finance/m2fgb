# M²FGB: A Min-Max Gradient Boosting Framework for Subgroup Fairness

In recent years, fairness in machine learning has emerged as a critical concern to ensure that developed and deployed predictive models do not have disadvantageous predictions for marginalized groups. It is essential to mitigate discrimination against individuals based on protected attributes such as gender and race. In this work, we consider applying subgroup justice concepts to gradient-boosting machines designed for supervised learning problems. Our approach expanded gradient-boosting methodologies to explore a broader range of objective functions, which combines conventional losses such as the ones from classification and regression and a min-max fairness term. We study relevant theoretical properties of the solution of the min-max optimization problem. The optimization process explored the primal-dual problems at each boosting round. This generic framework can be adapted to diverse fairness concepts. The proposed min-max primal-dual gradient boosting algorithm was theoretically shown to converge under mild conditions and empirically shown to be a powerful and flexible approach to address binary and subgroup fairness.


## Overview

This repository contains the implementation of M²FGB and executed experiments. This branch was updated with a improved implementation. To run the same code from the paper, please check the branch `facct25`.

## Installation

The recommend way to run the code is to set a Docker container. The file `Dockerfile`contains the configuration of the container utilized during development. Another way is to have Python installed on your machine. It is recommended to use a virtual environment to manage dependencies. Follow the steps below to set up your environment:

1. Create a virtual environment:
   ```bash
   python3 -m venv venv
   ```

2. Activate the virtual environment. On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
    
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Directory Structure

- `m2fgb/`: Contains Python scripts model implementation (`m2fgb.py` and `utils.py`) and evaluation (`evaluate.py`).
- `examples/`: Contains Jupyter notebooks with examples of the usage of models and visualizations.

## Quick Start

```python
from m2fgb import M2FGBClassifier
model = M2FGBClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=0,
)
model.fit(X_train, Y_train, A_train)
# A_train is the array of protected attributes
```

## Usage

wip

## Reference

For a detailed reference of the M²FGB framework, please refer to the [API documentation](reference.md).

## Citation

To cite this work, please use the following BibTeX entry:

```bibtex
@inproceedings{pereira2025m2fgb,
  title={M$^2$FGB: A Min-Max Gradient Boosting Framework for Subgroup Fairness},
  author={Pereira, Jansen Silva de Brito and Valdrighi, Giovani and Raimundo, Marcos Medeiros},
  booktitle={Proceedings of the 2025 ACM Conference on Fairness, Accountability, and Transparency},
  pages={3106--3118},
  year={2025}
}
```