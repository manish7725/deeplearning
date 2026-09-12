# Lab 17 — Assemble a Tiny Transformer Block

Start with embeddings and attention, then add a feed-forward network.

```python
import torch
import torch.nn as nn

x = torch.randn(2, 4, 8)  # batch, sequence, embedding
attn = nn.MultiheadAttention(embed_dim=8, num_heads=2, batch_first=True)

h, weights = attn(x, x, x)
h = h + x

ff = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 8))
out = ff(h) + h

print(out.shape)
print(weights.shape)
```

### Challenges
1. Change sequence length.
2. Change the number of heads.
3. Add layer normalization.
4. Explain why residual connections help information flow.
5. Sketch the block before running it.

### Resources
3Blue1Brown's transformer and attention lessons are the visual foundation. ZacharyLLM is the modern systems companion.