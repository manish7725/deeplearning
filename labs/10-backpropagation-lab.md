# Lab 10 — Backpropagation by Hand

Build a tiny chain:

`x → z=wx → a=z^2 → L=(a-y)^2`

```python
x, w, y = 2.0, 3.0, 10.0
z = w * x
a = z ** 2
L = (a - y) ** 2

# chain rule
dL_da = 2 * (a - y)
da_dz = 2 * z
dz_dw = x

dL_dw = dL_da * da_dz * dz_dw
print(L, dL_dw)
```

### Challenges
1. Compute the same gradient with finite differences.
2. Change `x`, `w`, and `y`.
3. Draw the computational graph by hand.
4. Reproduce it with `torch.autograd`.

### Resources
3Blue1Brown has dedicated lessons on backpropagation and its calculus. PyTorch's autograd documentation explains the computational graph and automatic differentiation.