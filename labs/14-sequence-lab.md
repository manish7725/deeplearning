# Lab 14 — Learn a Sequence State

Build a tiny recurrent update.

```python
import numpy as np

Wx = np.array([[0.5]])
Wh = np.array([[0.8]])
b = np.array([0.0])

h = np.array([0.0])
for x in [1., 2., 3., 4.]:
    h = np.tanh(Wx @ np.array([x]) + Wh @ h + b)
    print(h)
```

### Challenges
1. Change the sequence order and compare the final state.
2. Make `Wh` larger and observe memory behavior.
3. Explain why long sequences can create vanishing/exploding gradients.
4. Compare the idea with an LSTM cell.

### Resources
Use Frame Zero for foundational explanations and 3Blue1Brown for mathematical intuition. For modern sequence models, ZacharyLLM becomes more relevant as the series reaches LLMs.