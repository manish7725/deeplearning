# Lab 09 — Gradient Descent

Minimize `f(w)=(w-3)^2`.

```python
w = 0.0
lr = 0.1

for step in range(30):
    loss = (w - 3)**2
    grad = 2 * (w - 3)
    w -= lr * grad
    print(step, w, loss)
```

### Challenges
1. Try learning rates `0.01`, `0.1`, `0.5`, and `1.1`.
2. Find when training becomes unstable.
3. Plot `w` and loss over time.
4. Extend the experiment to learn both `w` and `b` in `y=wx+b`.

### Mastery question
What does the gradient tell us, and why does subtracting it usually move us downhill?