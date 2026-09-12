# Lab 02 — Explore Vectors

```python
import numpy as np

x = np.array([2., 3.])
w = np.array([4., 5.])

print("addition:", x + w)
print("scaling:", 2 * x)
print("dot product:", x @ w)
print("norm:", np.linalg.norm(x))
print("distance:", np.linalg.norm(x - w))
```

### Challenges
1. Create three student vectors and find the closest pair.
2. Normalize each vector to length 1.
3. Compare Euclidean distance with cosine similarity.
4. Explain why `[9, 8]` and `[8, 9]` are different vectors.

### Mastery question
What information is lost when a real-world object is converted into a vector?

### Resources
3Blue1Brown's linear-algebra lessons are especially useful for visualizing vectors, linear combinations and transformations. Use MrJensenMath10 for additional math practice and Frame Zero for ML intuition.