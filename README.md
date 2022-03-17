# Anderson Acceleration for Fixed-Point Iteration
Implementation of the (regularized) Anderson acceleration (aka Approximate Maximum Polynomial Extrapolation -- AMPE).

```python
>>> import numpy as np
>>> from aa import AndersonAcceleration
>>> acc = AndersonAcceleration(window_size=2, reg=0)
>>> x = np.random.rand(100)  # some iterate
>>> x_acc = acc.apply(x)     # accelerated from x
```

Check the jupyter notebook for examples.

<img src="AA_GD_quadratic.png" title="Comparing GD to AA on a quadratic objective with very high condition number">

TODO:
- [ ] Add an example for non-quadratic objective, e.g., logistic regression
- [ ] Add a version for `torch`'s model parameters