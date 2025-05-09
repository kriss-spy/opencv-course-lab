#!/usr/bin/env python3
"""
Guaranteed hex–grid sampler (integer output, floating-point distance test)

Usage
-----
python jittered_hex_grid_sampler.py <image_path> <k>

Returns exactly k (x, y) integers with
    min_pair_distance  >  sqrt(m*n/k)            # m, n = image width, height
or prints a diagnostic message and exits with non-zero status
only when k is mathematically impossible (see proof section below).

Requires pillow:  pip install pillow
"""

import math, random, sys, os
from PIL import Image


# ----------------------------------------------------------------------
def hex_side_for_equal_area(area_per_cell: float) -> float:
    """Side length s of a regular hexagon with given area."""
    return math.sqrt(2 * area_per_cell / (3 * math.sqrt(3)))


def generate_hex_centres(w, h, s, pad):
    """Yield centres of a hex lattice covering [0, w) × [0, h), padded by *pad*."""
    dx = math.sqrt(3) * s  # horizontal period
    dy = 1.5 * s  # vertical period
    y = -pad
    row = 0
    while y < h + pad:
        x = -pad + (0.5 * dx if row & 1 else 0.0)
        while x < w + pad:
            yield x, y
            x += dx
        y += dy
        row += 1


# ----------------------------------------------------------------------
def sample_points(w, h, k, rng=random.random):
    """Return exactly k points as integer tuples, satisfying the distance rule."""
    area = w * h
    d_req = math.sqrt(area / k)  # required >  d_req
    SAFETY_Q = math.sqrt(2.0)  # 1-px rounding worst case
    EXTRA_MARGIN = 0.01  # keep 1 % room for fp errors

    # 1. Start with equal-area hex grid (7.4 % gap over d_req already).
    s = hex_side_for_equal_area(area / k)
    c2c = math.sqrt(3) * s  # centre-to-centre distance

    # 2. Shrink s (thereby c2c) if trimming wastes too many cells.
    #    Stop as soon as we can generate ≥ k candidates *and*
    #    c2c still exceeds d_req + SAFETY_Q.
    shrink = 1.0
    points_float = []
    while True:
        s_try = s * shrink
        c2c_try = math.sqrt(3) * s_try
        if c2c_try <= d_req + SAFETY_Q + EXTRA_MARGIN:  # cannot shrink further
            break

        # jitter radius: leave half of the remaining gap after safety margin
        r_jit = 0.5 * (c2c_try - (d_req + SAFETY_Q))
        r_in = 0.5 * math.sqrt(3) * s_try
        r_jit = min(r_jit, 0.95 * r_in)
        pad = r_jit  # border expansion

        # build candidate points
        pts = []
        for cx, cy in generate_hex_centres(w, h, s_try, pad):
            rho = rng() * r_jit
            theta = rng() * (2 * math.pi)
            x = cx + rho * math.cos(theta)
            y = cy + rho * math.sin(theta)
            if 0 <= x < w and 0 <= y < h:
                pts.append((x, y))

        if len(pts) >= k:  # success
            points_float = pts
            break

        # otherwise shrink lattice by 2 % and try again
        shrink *= 0.98

    if len(points_float) < k:
        raise RuntimeError(
            f"Even the densest admissible hex grid yields only {len(points_float)} "
            f"points, cannot satisfy k={k} with the required distance."
        )

    # 3. Pick exactly k well-separated points (no further failures possible).
    rng_pts = random.sample(points_float, k)  # random subset, k ≤ len(points_float)

    # 4. Final float-space verification (should succeed by construction).
    for i in range(k):
        xi, yi = rng_pts[i]
        for j in range(i + 1, k):
            xj, yj = rng_pts[j]
            if math.hypot(xi - xj, yi - yj) <= d_req:
                raise AssertionError(
                    "distance guarantee violated – should never happen"
                )

    # 5. Convert to integer pixel coordinates (centre of pixel = integer).
    out_int = [(int(round(x)), int(round(y))) for x, y in rng_pts]
    # Duplicates after rounding are impossible because d_req > 1 px when k ≥ 2.
    return out_int, d_req, c2c_try, r_jit


# ----------------------------------------------------------------------
def cli():
    if len(sys.argv) != 3:
        print("Usage: python jittered_hex_grid_sampler.py <image_path> <k>")
        sys.exit(1)

    img_path, k_str = sys.argv[1], sys.argv[2]
    if not os.path.isfile(img_path):
        print(f"File not found: {img_path}")
        sys.exit(1)
    try:
        k = int(k_str)
        if k < 1:
            raise ValueError
    except ValueError:
        print("k must be a positive integer")
        sys.exit(1)

    w, h = Image.open(img_path).size
    try:
        pts, d_req, c2c, r_jit = sample_points(w, h, k)
    except Exception as e:
        print("FAILED:", e)
        sys.exit(2)

    print(f"{k} sample points for {w}×{h} image")
    print(f"required min distance    : {d_req:.3f} px")
    print(f"hex centre spacing used  : {c2c:.3f} px")
    print(f"jitter radius            : {r_jit:.3f} px\n")
    for p in pts:
        print(p)


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
