# Lab 15 — Build a Tiny Embedding Space

```python
import numpy as np

words = {
    "cat": np.array([1., 0., 0.]),
    "kitten": np.array([0.9, 0.1, 0.]),
    "car": np.array([0., 1., 0.]),
}

def cosine(a, b):
    return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))

for a in words:
    for b in words:
        print(a, b, round(cosine(words[a], words[b]), 3))
```

### Challenges
1. Add five words.
2. Create a simple 2D embedding plot.
3. Explain why cosine similarity ignores vector magnitude.
4. Experiment with a negative coordinate.

### Resources
3Blue1Brown's neural-network/LLM lessons provide visual context for representations. ZacharyLLM is a useful modern companion when embeddings become part of LLM systems.