# Lab 16 — Implement Self-Attention

```python
import torch

Q = torch.tensor([[1.,0.],[0.,1.]])
K = torch.tensor([[1.,0.],[0.,1.]])
V = torch.tensor([[10.,0.],[0.,20.]])

scores = Q @ K.T / (K.shape[-1] ** 0.5)
weights = torch.softmax(scores, dim=-1)
out = weights @ V

print("scores:\n", scores)
print("weights:\n", weights)
print("output:\n", out)
```

### Challenges
1. Change `Q` so one token prefers the second token.
2. Print row sums of the attention weights.
3. Add a causal mask.
4. Implement the same calculation with NumPy.

### Mastery question
Why does attention need separate query, key and value representations?

### Resources
3Blue1Brown now has dedicated lessons on attention and transformers. ZacharyLLM is a useful companion for modern LLM systems.