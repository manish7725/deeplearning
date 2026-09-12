# Lab 08 — Numerical Derivatives

For `f(x)=x^2`, the derivative is `2x`. Verify it numerically.

```python

def f(x):
    return x**2

def numerical_derivative(f, x, h=1e-5):
    return (f(x+h) - f(x-h)) / (2*h)

for x in [-2., -1., 0., 1., 2.]:
    print(x, numerical_derivative(f, x), 2*x)
```

### Challenges
1. Try `f(x)=x**3`.
2. Try `f(x)=sin(x)`.
3. Reduce `h` and observe numerical error.
4. Explain the geometric meaning of the derivative as slope.

### Resources
3Blue1Brown's Essence of Calculus is the primary visual companion. MrJensenMath10 is a useful additional source for calculus practice.