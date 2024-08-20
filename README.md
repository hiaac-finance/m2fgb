# M²FGB: A Min-Max Gradient Boosting Framework for Subgroup Fairness

In recent years, fairness in machine learning has emerged as a critical concern to guarantee that developed and deployed predictive models do not have distinct predictions for marginalized groups. However, it is essential to avoid biased decisions and promote equitable outcomes dealing with multiple (sub)group attributes (gender, race, etc.) simultaneously. In this work, we consider applying subgroup justice concepts to gradient-boosting machines designed for supervised learning problems, specifically binary classification. Our approach expanded gradient boosting methodologies to explore a broader range of objective functions using min-max formulation exploring primal-dual optimization. This generic framework can be adapted to diverse fairness concepts. The proposed min-max primal-dual gradient boosting algorithm was empirically shown to be a powerful and flexible approach to address binary and subgroup fairness.


## Overview

This repository contains the implementation of M²FGB and executed experiments. The `scripts` directory contains Python scripts for data processing, model training, and evaluation, while the `notebooks` directory contains Jupyter notebooks with usage examples and visualization.

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

- `scripts/`: Contains Python scripts model implementation (`models.py`), evaluation (`utils.py`), data preprocessing (`preprocess_data.py`and `data.py`) and the exeucted experiments (`experiments.py`).
- `notebooks/`: Contains Jupyter notebooks with examples of the usage of models and visualizations.
- `data/`: A directory with preprocessed datasets.
- `minimax-fair/` and `MMPF/` are clones of the github repository of related techniques.

## Usage

1. **Experiments**:

    ```bash
     python scripts/experiments.py
     ```

     You can add the flag `--experiment reg` to run the regression experiment or `--experiment fair_weight` to run the experiment that eval the `fair_weight` parameter.

2. **M²FGB**: Example usage of the propposed technique is present at `notebooks/usage_m2fgb.ipynb`.