import matplotlib
matplotlib.use("tkagg")

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(projection="3d")

C = [[0, 0], [0, 3/8], [0, 1], [1, 0], [1, 2/8], [1, 3/8], [1, 1]]
V = [0, 1, 0, 0, 1, 1, 0]

kernel = RBF()
gpr = GaussianProcessRegressor(kernel=kernel, random_state=0, n_restarts_optimizer=10).fit(C, V)

SX = np.linspace(0.0, 1.0, 100)
SY = np.linspace(0.0, 1.0, 100)

MX, MY = np.meshgrid(SX, SY)

MZ = np.array(gpr.predict([[a, b] for a in SX for b in SY])).reshape(MX.shape)

ax.plot_surface(MX, MY, MZ)
# ax.set_aspect("equal")

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

for i, c in enumerate(C):
    v = V[i]
    predicted = gpr.predict([c])[0]
    print(f"Error for point ({c}, {v}) (predicted: {predicted}): (v[i] - v*[i])^2 = {(v - predicted) ** 2}")

plt.show()
