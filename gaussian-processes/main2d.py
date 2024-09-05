import matplotlib
matplotlib.use("tkagg")

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

import numpy as np
import matplotlib.pyplot as plt

X = [[0], [1]]
y = [1, 0]

kernel = RBF()
gpr = GaussianProcessRegressor(kernel=kernel).fit(X, y)

SX = np.linspace(0.0, 1.0, 100)
SY = gpr.predict([[x] for x in SX])

print(SY)

plt.plot(SX, SY)
plt.show()
