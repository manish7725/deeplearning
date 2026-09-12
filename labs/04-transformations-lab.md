# Lab 04 — Transform a Shape

```python
import numpy as np

square = np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])

scale = np.array([[2.,0.],[0.,0.5]])
rotate90 = np.array([[0.,-1.],[1.,0.]])
reflect_x = np.array([[1.,0.],[0.,-1.]])

print(square @ scale.T)
print(square @ rotate90.T)
print(square @ reflect_x.T)
```

### Challenges
1. Build a shear matrix.
2. Rotate by 180 degrees.
3. Compare `A @ B` with `B @ A`.
4. Add a bias vector and observe the translation.

### Mastery question
Why is `Wx+b` affine rather than strictly linear when `b != 0`?

### Resources
3Blue1Brown's Essence of Linear Algebra is the visual companion. Use Welch Labs when you want a story-driven explanation of why mathematical transformations become useful in AI.