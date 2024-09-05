import pygame
import numpy as np

points = np.array([
    [0, 0], [0, 3/8], [0, 1], [1, 0], [1, 2/8], [1, 3/8], [1, 1]
])

k = len(points)

values = np.array([0, 1, 0, 0, 1, 1, 0])

def sim(x, y):
    epsilon = 0.5
    return np.exp(- epsilon * np.dot(y - x, y - x))

interp = np.array([
    [sim(points[j], points[i]) for j in range(k)]
    for i in range(k)
])

weights = np.linalg.solve(interp, values)

def rbf(v):
    return np.clip(
        np.dot(weights, np.array([sim(v, point) for point in points])),
        0, 1
    )


# Size of window, number of pixels, width of pixels
S = 600
L = 200
W = S // L

def average_neighbor(buffer, y, x):
    total = sum(buffer[(y + dy + L) % L][(x + dx + L) % L] for dy in [-1, 0, 1] for dx in [-1, 0, 1]) - buffer[y][x]
    return total / 8

# buffer = np.array([[float((i + j) % 2) for i in range(L)] for j in range(L)]) 
buffer = np.random.rand(L, L)

pygame.init()
screen = pygame.display.set_mode((S, S))
clock = pygame.time.Clock()

running = True

i = 0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    back_buffer = np.zeros((L, L))

    for y in range(L):
        for x in range(L):
            value = int(255 * buffer[y][x])
            pygame.draw.rect(screen, (value, value, value), (x * W, y * W, W, W))

            back_buffer[y][x] = rbf(np.array([buffer[y][x], average_neighbor(buffer, y, x)]))

    np.copyto(buffer, back_buffer)

    pygame.display.flip()

    # pygame.image.save(pygame.display.get_surface(), f"frames/frame{i:03}.png")

    clock.tick(5)

    i += 1

pygame.quit()
