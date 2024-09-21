from PIL import Image
import numpy as np

N = 600

points = np.array([
    [0, 0], [0, 2/8], [0, 5/8], [0, 1], [1, 0], [1, 2/8], [1, 3/8], [1, 1]
])

k = len(points)

values = np.array([0, 0.5, 1, 0, 0, 1, 1, 0])

def sim(x, y):
    epsilon = 3.190
    return np.exp(- epsilon * np.dot(y - x, y - x))

interp = np.array([
    [sim(points[j], points[i]) for j in range(k)]
    for i in range(k)
])

weights = np.linalg.solve(interp, values)

def rbf(v):
    return np.clip(sum(weights[i] * sim(v, points[i]) for i in range(k)), 0, 1)

def value_to_gray(t):
    return (int(255 * t), int(255 * t), int(255 * t))

rbfs = [
    [rbf(np.array([j / (N - 1), i / (N - 1)])) for j in range(N)]
    for i in range(N)
]

change = sum([
    sum([rbfs[i][j] - j / (N - 1) for j in range(N)])
    for i in range(N)
])

print("Total change:")
print(change)

pixels = [
    [value_to_gray(rbfs[i][j]) for j in range(N)]
    for i in range(N)
]

data = np.array(pixels, dtype=np.uint8)

img = Image.fromarray(data)
img.save("bin/rbf.png")
