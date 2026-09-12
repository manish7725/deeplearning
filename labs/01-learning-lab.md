# Lab 01 — Build a Tiny Learner

## Goal
Make a model learn the rule `y = 2x + 1` from examples.

```python
import numpy as np

x = np.array([1., 2., 3., 4.])
y = np.array([3., 5., 7., 9.])

w, b = 0.0, 0.0
lr = 0.01

for step in range(1000):
    pred = w * x + b
    error = pred - y
    loss = np.mean(error ** 2)

    dw = np.mean(2 * error * x)
    db = np.mean(2 * error)

    w -= lr * dw
    b -= lr * db

print(w, b, loss)
```

### Challenges
1. Change the target to `y = 3x - 2`.
2. Try three different learning rates.
3. Print the loss every 100 steps.
4. Plot prediction versus truth.

### Mastery question
Why do `w` and `b` need to change if the model already knows the formula `wx+b`?

### Resources
Use Frame Zero, Welch Labs and 3Blue1Brown for intuition; use the PyTorch beginner tutorials when you are ready to reproduce the same learning loop with tensors and autograd.