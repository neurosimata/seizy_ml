### Customizing Models and Hyperparameters in SeizyML

SeizyML allows users to tweak hyperparameters of the **Gaussian Naive Bayes (GNB)** model or even experiment with different machine learning models. This guide explains how to modify these settings for custom experimentation.

---

####  Modifying Hyperparameters for Gaussian Naive Bayes (GNB)

The key hyperparameter for GNB is `var_smoothing`, which controls the stability of variance calculations.

**Steps to Modify:**
1. Open the `model_settings.py` file in the SeizyML repository.
2. Locate the following section:

```python
from sklearn.naive_bayes import GaussianNB
import numpy as np

# Model and hyperparameters
define_metrics = 'BALANCED_ACCURACY'
metrics = {'AUC': 'roc_auc', 'BALANCED_ACCURACY': 'balanced_accuracy', 'PRECISION': 'precision', 'RECALL': 'recall', 'F1': 'f1'}

models = {'gaussian_nb': GaussianNB()}

hyper_params = {
    'gaussian_nb': {'var_smoothing': np.logspace(-2, -8, num=7)}
}
```

3. Adjust the `var_smoothing` range to experiment with different values:

```python
hyper_params = {
    'gaussian_nb': {'var_smoothing': np.logspace(-1, -9, num=9)}  # Adjusted range
}
```

4. Save the file and re-run the model training process.

---

#### Adding Custom Models

You can integrate other machine learning models like **Support Vector Machines (SVM)** or **Logistic Regression** implemented through the SGD model from scikit-learn.

**Example, adding an SGD classifier:**

1. Import the desired model
```python
from sklearn.linear_model import SGDClassifier
```

2. Add the model to the `models` dictionary
```python
models = {
    'gaussian_nb': GaussianNB(),
    'sgd': SGDClassifier()
}
```

3. Define hyperparameters for the new model
```python
hyper_params = {
    'gaussian_nb': {'var_smoothing': np.logspace(-2, -8, num=7)},
    'sgd': {
        'loss': ['hinge', 'log_loss'],
        'alpha': [0.0001, 0.001, 0.01],
        'penalty': ['l2', 'l1', 'elasticnet'],
        'max_iter': [1000, 2000]
    }
}
```

4. Save the changes and execute the model training pipeline.

---
**[<< Back to Main Page](/README.md)**