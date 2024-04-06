
import sys
sys.path.append("../scripts")

import experiments


fairness_metric = "min_bal_acc"
thresh = "ks"

for dataset in ["german", "adult"]:
    for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        for model_name in ["M2FGB_grad"]:
            args = {
                "dataset": dataset,
                "alpha": alpha,
                "output_dir": f"../results/comparing_metrics/{dataset}/{model_name}_{alpha}_{fairness_metric}",
                "model_name": model_name,
                "n_trials": 10,
                "n_groups": 4,
                "thresh" : thresh,
                "fairness_metric": fairness_metric,
            }
            experiments.run_subgroup_experiment(args)