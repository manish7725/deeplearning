# Lab 18 — Train a Tiny Next-Token Model

Use a tiny vocabulary and learn the next token in a sequence.

```python
import torch
import torch.nn as nn

vocab = 5
model = nn.Sequential(
    nn.Embedding(vocab, 8),
    nn.Flatten(),
    nn.Linear(8 * 3, vocab)
)

x = torch.tensor([[0,1,2], [1,2,3]])
y = torch.tensor([3,4])

logits = model(x)
loss = nn.CrossEntropyLoss()(logits, y)
print(logits.shape, loss.item())
```

### Challenges
1. Train the model for 500 steps.
2. Print the predicted next token.
3. Change the training sequences.
4. Explain why the final layer outputs one logit per vocabulary item.

### Resources
3Blue1Brown's LLM/transformer lessons give conceptual intuition. ZacharyLLM is a useful resource for modern LLM architecture and systems.