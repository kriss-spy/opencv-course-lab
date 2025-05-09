#!/usr/bin/env python3
"""
Buffered jittered-hex-grid sampler

• Uses a dense regular hex lattice (equal–area cells)
• Enlarges the lattice so some cell centres sit *outside* the image;
  this lets edge cells keep their full jitter disk (“border expansion”)
• Guarantees   min_pair_distance  >  √(m·n / k)   even after
  integer-pixel rounding (uses a 1.414 px quantisation safety margin)
• CLI usage:  python jittered_hex_grid_sample.py  <image_path>  <k>
"""

import math, random, sys, os
from PIL import Image


# ----------------------------------------------------------------------
def build_params(w: int, h: int, k: int):
    area = w * h
    min_dist = math.sqrt(area / k)  # distance required by the task

    # hexagon with area = area/k   ⇒ side length s
    s = math.sqrt(2 * area / (3 * math.sqrt(3) * k))
    d_centres = math.sqrt(3) * s  # closest lattice spacing

    # extra cushion so rounding to the nearest pixel cannot break the guarantee
    safety = math.sqrt(2)  # ≈ 1.414 px worst-case shrink
    r_jit = max(0.0, (d_centres - (min_dist + safety)) / 2)

    r_in = 0.5 * math.sqrt(3) * s  # hex in-radius
    r_jit = min(r_jit, 0.95 * r_in)  # stay inside the hex

    if r_jit == 0.0:
        raise ValueError(
            "Image is too small or k too large to satisfy the distance constraint."
        )

    params = {
        "s": s,
        "d_centres": d_centres,
        "r_jit": r_jit,
        "min_dist": min_dist,
        "pad": r_jit,  # buffer added outside the image
    }
    return params


# ----------------------------------------------------------------------
def jittered_hex_sample(w: int, h: int, k: int):
    p = build_params(w, h, k)
    s, r_jit, pad = p["s"], p["r_jit"], p["pad"]

    # Step vectors of the axial lattice
    dx = math.sqrt(3) * s  # horizontal step
    dy = 1.5 * s  # vertical step

    points = []
    row = 0
    y = -pad
    # vertical sweep including top/bottom buffer
    while y <= h + pad:
        # staggering: every odd row shifts right by dx/2
        x_offset = 0.0 if row % 2 == 0 else 0.5 * dx
        x = -pad + x_offset
        # horizontal sweep including left/right buffer
        while x <= w + pad:
            # random polar offset inside jitter disk
            rho = random.uniform(0, r_jit)
            theta = random.uniform(0, 2 * math.pi)
            px = x + rho * math.cos(theta)
            py = y + rho * math.sin(theta)
            if 0.0 <= px < w and 0.0 <= py < h:
                points.append((px, py))
            x += dx
        y += dy
        row += 1

    if len(points) < k:
        raise RuntimeError(
            f"Only produced {len(points)} candidates, need k={k}. "
            "Try smaller k or a larger image."
        )

    # Randomise and take first k
    random.shuffle(points)
    points = points[:k]

    # Final sanity check in *float* space
    min_d = p["min_dist"]
    for i in range(k):
        xi, yi = points[i]
        for j in range(i + 1, k):
            xj, yj = points[j]
            if math.hypot(xi - xj, yi - yj) <= min_d:
                raise RuntimeError("Internal error: distance guarantee violated.")

    return points, p


# ----------------------------------------------------------------------
def cli():
    if len(sys.argv) != 3:
        print("Usage: python jittered_hex_grid_sample.py <image_path> <num_points>")
        sys.exit(1)

    img_path = sys.argv[1]
    if not os.path.isfile(img_path):
        print(f"File not found: {img_path}")
        sys.exit(1)
    try:
        k = int(sys.argv[2])
        if k <= 0:
            raise ValueError
    except ValueError:
        print("num_points must be a positive integer")
        sys.exit(1)

    w, h = Image.open(img_path).size
    try:
        pts, params = jittered_hex_sample(w, h, k)
    except Exception as e:
        print("FAILED:", e)
        sys.exit(1)

    # Convert to integer pixel coordinates *for display only*
    int_pts = {(int(round(x)), int(round(y))) for (x, y) in pts}

    print(f"Buffered jittered-hex sampling on {w}×{h} image")
    print(f"Requested points: {k}")
    print(f"Hex side s        : {params['s']:.3f}")
    print(f"Centre spacing    : {params['d_centres']:.3f}")
    print(f"Jitter radius     : {params['r_jit']:.3f}")
    print(f"Distance required : {params['min_dist']:.3f} px\n")
    print("Sampled (x, y) pixel positions:")
    for pt in sorted(int_pts):
        print(pt)
    if len(int_pts) < k:
        print(
            f"\nNote: {k - len(int_pts)} points collided after rounding to ints "
            "(duplicates removed). "
            "Increase image size or lower k if you need *distinct* pixel indices."
        )


# ----------------------------------------------------------------------
if __name__ == "__main__":
    cli()
