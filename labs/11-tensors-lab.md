# Lab 11 — Tensor Shapes

```python
import torch

x = torch.tensor([1,2,3])
X = torch.tensor([[1,2,3],[4,5,6]])
T = torch.zeros(2,3,4)

print(x.shape, X.shape, T.shape)
print("ndim:", T.ndim)
```

### Challenges
1. Create a tensor shaped `(batch, channels, height, width)`.
2. Test broadcasting with shapes `(3,1)` and `(1,4)`.
3. Predict the result shape before every operation.
4. Move a tensor to GPU when CUDA is available.

### Mastery question
Why are shape errors often more informative than numerical errors in deep learning?