import sys
import math
import random
import os
from PIL import Image


def hex_grid_jittered_sampling(w, h, k):
    # Compute cell and grid geometry
    if k <= 0 or w <= 0 or h <= 0:
        raise ValueError("Image dimensions and k must be positive")
    area = w * h
    if k > area:
        raise ValueError("Cannot place more points than pixels in the image")
    a_cell = area / k
    # Side of hexagon so that Area = a_cell
    s = math.sqrt(2 * area / (3 * math.sqrt(3) * k))
    d_centres = math.sqrt(3) * s  # Smallest distance between centres (~1.07 x desired)
    min_dist = math.sqrt(area / k)
    r_jit = (d_centres - min_dist) / 2.0
    # Defensive: Clamp jitter to never exceed inner radius
    r_in = (math.sqrt(3) / 2) * s
    r_jit = min(r_jit, r_in * 0.95, min_dist * 0.5)
    if r_jit <= 0:
        raise ValueError("Not enough room for required minimal spacing.")

    # Build hex lattice roughly covering the image
    y = 0
    points = []
    row_idx = 0
    while y < h:
        # Offset x for every other row
        x_offset = 0 if row_idx % 2 == 0 else 0.5 * s * math.sqrt(3)
        x = x_offset
        while x < w:
            # Jitter in allowed circle
            angle = random.uniform(0, 2 * math.pi)
            rad = random.uniform(0, r_jit)
            x_jit = x + rad * math.cos(angle)
            y_jit = y + rad * math.sin(angle)
            xi, yi = int(round(x_jit)), int(round(y_jit))
            if 0 <= xi < w and 0 <= yi < h:
                points.append((xi, yi))
            x += s * math.sqrt(3)
        y += 1.5 * s
        row_idx += 1

    # Remove possible accidental duplicates (due to rounding/jitter), keep first k only.
    uniq = []
    visited = set()
    for pt in points:
        if pt not in visited:
            uniq.append(pt)
            visited.add(pt)
            if len(uniq) == k:
                break

    # Now, verify all mutual distances
    def euclid(p, q):
        return math.hypot(p[0] - q[0], p[1] - q[1])

    too_close = []
    for i in range(len(uniq)):
        for j in range(i + 1, len(uniq)):
            if euclid(uniq[i], uniq[j]) < min_dist:
                too_close.append((uniq[i], uniq[j]))

    if len(uniq) < k:
        raise RuntimeError(
            f"Could only select {len(uniq)} unique points for k={k} (image too small or weird settings)"
        )
    if too_close:
        print("FAILED: point pairs are too close!")
        for a, b in too_close[:5]:
            print(
                f"  Points {a} and {b} are {euclid(a,b):.2f} apart, required >{min_dist:.2f}"
            )
        return None
    return uniq


def main():
    if len(sys.argv) < 3:
        print("Usage: python hex_jitter_sample.py <image_path> <num_points>")
        return
    img_path = sys.argv[1]
    k = int(sys.argv[2])
    if not os.path.isfile(img_path):
        print(f"Cannot find '{img_path}'")
        return
    img = Image.open(img_path)
    w, h = img.size
    try:
        pts = hex_grid_jittered_sampling(w, h, k)
    except Exception as e:
        print("FAILED:", e)
        return
    if pts is None:
        print("Sampling failed.")
    else:
        print(f"Sampled {len(pts)} points for {w}x{h} image with k={k}:")
        if len(pts) < 100:
            for pt in pts:
                print(pt)
        print(f"Minimum allowed pair distance: {math.sqrt(w*h/k):.2f}")


if __name__ == "__main__":
    main()
