# Lab 03 — Matrix Multiplication

```python
import torch

X = torch.tensor([[2., 3.], [4., 5.]])
W = torch.tensor([[10.], [20.]])

print(X @ W)
print("X shape:", X.shape)
print("W shape:", W.shape)
```

### Challenges
1. Predict every output before running the code.
2. Try `X @ torch.randn(3, 1)` and explain the error.
3. Create a `64 x 128` input and a `128 x 256` weight matrix.
4. Verify that the output shape is `64 x 256`.

### Mastery question
Why does matrix multiplication compute many dot products at once?

### Resources
3Blue1Brown's linear algebra series gives the geometric view of matrix multiplication. PyTorch's tensor tutorials provide the programming view.