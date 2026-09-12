# Lab 20 — Build the Whole Learner Yourself

Start with NumPy and then reproduce it in PyTorch.

```python
import numpy as np

x = np.array([[1.],[2.],[3.],[4.]])
y = 2*x + 1

W = np.random.randn(1,1)
b = np.zeros((1,1))
lr = 0.05

for step in range(1000):
    pred = x @ W + b
    err = pred - y
    loss = np.mean(err**2)

    dW = (2/x.shape[0]) * x.T @ err
    db = (2/x.shape[0]) * np.sum(err)

    W -= lr * dW
    b -= lr * db

print("W:", W)
print("b:", b)
print("loss:", loss)
```

### Final challenges
1. Add a hidden layer.
2. Add ReLU.
3. Derive every gradient before coding it.
4. Compare your gradients with PyTorch autograd.
5. Deliberately introduce a wrong gradient and diagnose the training failure.

### Mastery test
If you can explain the forward pass, loss, derivative, gradient update and shape of every tensor without looking at the code, you have completed the first-principles core of this series.

### Resources
Use PyTorch's official beginner tutorials for the framework implementation and 3Blue1Brown/Welch Labs for mathematical intuition. PyTorch explicitly provides beginner material covering tensors, autograd, neural networks and optimization.