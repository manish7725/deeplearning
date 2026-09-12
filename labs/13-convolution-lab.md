# Lab 13 — Convolution by Hand

Start with a small image and kernel.

```python
import numpy as np

image = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9]
])

kernel = np.array([
    [1,0],
    [0,-1]
])

out = np.zeros((2,2))
for i in range(2):
    for j in range(2):
        patch = image[i:i+2, j:j+2]
        out[i,j] = np.sum(patch * kernel)

print(out)
```

### Challenges
1. Change the kernel to an edge detector.
2. Implement stride 2.
3. Add padding.
4. Reproduce the operation with `torch.nn.Conv2d`.

### Mastery question
Why does a convolution kernel allow the same detector to search across an entire image?