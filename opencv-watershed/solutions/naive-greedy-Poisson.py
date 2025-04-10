import random
import math
import numpy as np


def generate_seeds(M, N, K):
    """
    Generate K seed points within an M x N matrix using Poisson Disk Sampling + Greedy Strategy.
    Ensures minimum distance between seeds is greater than (M * N / K)^0.5.

    :param M: Number of rows in the matrix (height of the image).
    :param N: Number of columns in the matrix (width of the image).
    :param K: Number of seeds to generate.
    :return: List of seed points [(x1, y1), (x2, y2), ...].
    """
    d = math.sqrt(M * N / K)  # Minimum required distance between seeds
    seeds = []

    attempts = 0  # Track number of attempts to place seeds
    max_attempts = K * 500  # Avoid infinite loops by limiting attempts
    # max_attempts = float("inf")

    while len(seeds) < K and attempts < max_attempts:
        # Generate a random point in the matrix
        x, y = random.randint(0, N - 1), random.randint(0, M - 1)

        # Check if the point is far enough from all existing seeds
        valid = True
        for sx, sy in seeds:
            if math.sqrt((x - sx) ** 2 + (y - sy) ** 2) < d:
                valid = False
                break

        if valid:
            seeds.append((x, y))  # Add the valid seed
        attempts += 1

    # Check if the algorithm succeeded
    if len(seeds) == K:
        return seeds
    else:
        raise Exception(
            f"Failed to generate {K} seeds after {max_attempts} attempts. Try reducing K or increasing matrix size."
        )


def verify_seeds(matrix_shape, K, seeds):
    """
    Verify the generated seed points to ensure they satisfy the minimum distance condition.

    :param matrix_shape: Tuple (M, N) representing matrix dimensions.
    :param K: Number of seeds required.
    :param seeds: List of seed points [(x1, y1), (x2, y2), ...].
    :return: True if seeds are valid, otherwise False.
    """
    M, N = matrix_shape
    d = math.sqrt(M * N / K)  # Minimum required distance

    # Verify distances between every pair of seeds
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            sx1, sy1 = seeds[i]
            sx2, sy2 = seeds[j]
            distance = math.sqrt((sx2 - sx1) ** 2 + (sy2 - sy1) ** 2)
            if distance < d:
                print(
                    f"Seed points {seeds[i]} and {seeds[j]} are too close (distance = {distance:.2f})."
                )
                return False

    return True


if __name__ == "__main__":
    # Input parameters
    M = 600  # Matrix height
    N = 600  # Matrix width
    K = 50  # Number of seeds
    # always fail
    # bad algorithm

    print(f"Generating {K} seeds for a {M}x{N} matrix...")

    try:
        # Step 1: Generate seeds
        seeds = generate_seeds(M, N, K)
        print(f"Generated {len(seeds)} seeds successfully.")

        # Step 2: Verify the seeds
        is_valid = verify_seeds((M, N), K, seeds)
        if is_valid:
            print("All seeds satisfy the minimum distance requirement.")
            print("Seed points:", seeds)
        else:
            print("Some seeds do not satisfy the minimum distance condition.")
    except Exception as e:
        print(f"Error: {e}")
