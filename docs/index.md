# Introduction

## Installation


## Quick Start

```python
from m2fgb import M2FGBClassifier
model = M2FGBClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=0,
)
model.fit(X_train, Y_train, A_train)
```

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