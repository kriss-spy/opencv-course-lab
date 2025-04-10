# random_jittered_grid.py
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, ceil


def jittered_grid_sampling(
    M: int, N: int, K: int, seed: int = None, temperature: float = 1
):
    if seed is not None:
        np.random.seed(seed)

    # Calculate grid dimensions based on aspect ratio
    aspect_ratio = N / M
    r = int(round(sqrt(K / aspect_ratio)))  # Number of rows
    c = int(ceil(K / r))  # Number of columns

    # Calculate cell dimensions
    cell_height = M / r
    cell_width = N / c

    points = []

    for i in range(r):
        for j in range(c):
            # Place points at center of each cell plus jitter
            y = int(
                (i + 0.5) * cell_height
                + temperature * np.random.uniform(-0.4 * cell_height, 0.4 * cell_height)
            )
            x = int(
                (j + 0.5) * cell_width
                + temperature * np.random.uniform(-0.4 * cell_width, 0.4 * cell_width)
            )

            # Ensure points stay within bounds
            y = max(0, min(M - 1, y))
            x = max(0, min(N - 1, x))

            points.append((x, y))

    # Return exactly K points
    return points[:K]


# --- DEMO ---
if __name__ == "__main__":
    M, N, K = 600, 600, 100
    points = jittered_grid_sampling(M, N, K, seed=45, temperature=0)

    plt.figure(figsize=(8, 6))
    xs, ys = zip(*points)
    plt.scatter(xs, ys, s=10, color="blue")
    plt.xlim(0, N)
    plt.ylim(0, M)
    plt.gca().invert_yaxis()
    plt.title(f"Jittered Grid Sampling (K={K})")
    plt.show()
