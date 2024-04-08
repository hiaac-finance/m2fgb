
import sys
sys.path.append("../scripts")

import experiments


fairness_metric = "min_bal_acc"
thresh = "ks"

for dataset in ["german", "adult"]:
    for alpha in [0, 0.2, 0.4, 0.6, 0.8, 1]:
        for model_name in ["M2FGB", "M2FGB_grad", "FairGBMClassifier"]:
            args = {
                "dataset": dataset,
                "alpha": alpha,
                "output_dir": f"../results/comparing_metrics_2/{dataset}/{model_name}_{alpha}_{fairness_metric}",
                "model_name": model_name,
                "n_trials": 100,
                "n_folds" : 5,
                "n_groups": 4,
                "thresh" : thresh,
                "fairness_metric": fairness_metric,
            }
            experiments.run_subgroup_experiment(args)