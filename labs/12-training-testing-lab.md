# Lab 12 — Train, Validate, Test

Create synthetic data and split it.

```python
from sklearn.model_selection import train_test_split
import numpy as np

X = np.arange(200).reshape(100, 2)
y = X[:, 0] + 2 * X[:, 1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(X_train.shape, X_test.shape)
```

### Challenges
1. Add a validation split.
2. Intentionally overfit a high-degree polynomial.
3. Compare training and validation error.
4. Explain why the test set should remain untouched until the end.

### Resources
The scikit-learn User Guide is a strong reference for model selection and evaluation. PyTorch's beginner workflow covers datasets, training and evaluation.