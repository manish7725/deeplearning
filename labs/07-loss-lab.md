# Lab 07 — Build a Loss Function

```python
import numpy as np

y = np.array([3., 5., 7., 9.])
pred = np.array([2., 6., 6., 10.])

mse = np.mean((pred - y) ** 2)
mae = np.mean(np.abs(pred - y))

print("MSE:", mse)
print("MAE:", mae)
```

### Challenges
1. Change one prediction at a time and observe the loss.
2. Compare MSE and MAE for an outlier.
3. Implement binary cross-entropy for one probability prediction.
4. Explain why a loss function must be a number that can be optimized.

### Mastery question
Can a model have good training loss but poor real-world performance? Explain why.