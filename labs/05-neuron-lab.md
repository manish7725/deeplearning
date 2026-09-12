# Lab 05 — Implement a Neuron

```python
import numpy as np

x = np.array([2., 3.])
w = np.array([4., 5.])
b = 1.

z = w @ x + b
print(z)
```

Now create three neurons:

```python
W = np.array([[4.,5.],[1.,2.],[-2.,3.]])
b = np.array([1.,0.,2.])
print(W @ x + b)
```

### Challenges
1. Add a third input feature.
2. Make one weight negative and predict the effect.
3. Reproduce the calculation in PyTorch with `torch.nn.Linear`.
4. Explain every dimension in the matrix multiplication.

### Mastery question
Why is a neuron better understood as a mathematical function than as a tiny brain?

### Resources
3Blue1Brown's neural-network series and Welch Labs' Neural Networks Demystified are excellent companions for neuron architecture and forward propagation.