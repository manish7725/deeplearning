# Lab 06 — See Why Nonlinearity Matters

```python
import numpy as np

z = np.linspace(-5, 5, 11)
relu = np.maximum(0, z)
sigmoid = 1 / (1 + np.exp(-z))

tanh = np.tanh(z)

print(np.c_[z, relu, sigmoid, tanh])
```

### Challenges
1. Implement ReLU without NumPy's `maximum`.
2. Implement sigmoid from scratch.
3. Compare `A2 @ (A1 @ x)` with two layers containing ReLU.
4. Plot all three activation functions.

### Mastery question
Why can't stacking many purely linear layers create the same expressive power as a network with nonlinear activations?

### Resources
3Blue1Brown discusses ReLU versus sigmoid in its neural-network series. Welch Labs provides visual intuition for neural-network behavior.