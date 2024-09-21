import pygame
import numpy as np

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
    return np.clip(
        np.dot(weights, np.array([sim(v, point) for point in points])),
        0, 1
    )


# Size of window, number of pixels, width of pixels
S = 600
L = 50
W = S // L

def buffer_index(buffer, y, x):
    return buffer[(y + L) % L][(x + L) % L]
    
    """
    if y < 0 or x < 0 or y >= L or x >= L:
        return 0.0
    return buffer[y][x]
    """
    

def average_neighbor(buffer, y, x):
    total = sum(buffer_index(buffer, y + dy, x + dx) for dy in [-1, 0, 1] for dx in [-1, 0, 1]) - buffer[y][x]
    return total / 8

# buffer = np.array([[float((i + j) % 2) for i in range(L)] for j in range(L)]) 
buffer = np.random.rand(L, L)

pygame.init()
screen = pygame.display.set_mode((S, S))
clock = pygame.time.Clock()

running = True
updating = False

i = 0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            updating = not updating
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_c:
            buffer = np.zeros((L, L))
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = pygame.mouse.get_pos()

            x, y = mx // W, my // W

            buffer[y][x] = min(1, (buffer[y][x] + 0.25) % 1.25)

    back_buffer = np.zeros((L, L))

    for y in range(L):
        for x in range(L):
            value = int(255 * buffer[y][x])
            pygame.draw.rect(screen, (value, value, value), (x * W, y * W, W, W))

            if updating:
                back_buffer[y][x] = rbf(np.array([buffer[y][x], average_neighbor(buffer, y, x)]))

    if updating:
        change = back_buffer - buffer
        print("Change: ", sum(change.ravel()))

        np.copyto(buffer, back_buffer)

    pygame.display.flip()

    # pygame.image.save(pygame.display.get_surface(), f"frames/frame{i:03}.png")

    clock.tick(5)

    if updating:
        i += 1

pygame.quit()
