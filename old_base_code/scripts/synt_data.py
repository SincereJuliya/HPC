import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

# ==========================================================
#   DATA GENERATORS
# ==========================================================

def gen_gaussian(n, centers, spread):
    """
    Generate Gaussian clusters.

    Args:
        n (int): Number of points per cluster.
        centers (list of tuple): List of (x, y) centers.
        spread (float): Standard deviation of clusters.

    Returns:
        X (np.ndarray): 2D coordinates of points.
        y (np.ndarray): Labels corresponding to clusters.
    """
    data = []
    labels = []
    for i, (cx, cy) in enumerate(centers):
        x = np.random.normal(cx, spread, n)
        y = np.random.normal(cy, spread, n)
        data.append(np.column_stack([x, y]))
        labels += [i] * n
    return np.vstack(data), np.array(labels)


def gen_two_moons(n, noise):
    """
    Generate a two-moons dataset.

    Args:
        n (int): Total number of points.
        noise (float): Standard deviation of Gaussian noise.

    Returns:
        X (np.ndarray): 2D coordinates of points.
        y (np.ndarray): Labels (0 or 1) for the moons.
    """
    n1 = n // 2
    t = np.linspace(0, np.pi, n1)
    x1 = np.cos(t) + np.random.normal(0, noise, n1)
    y1 = np.sin(t) + np.random.normal(0, noise, n1)

    x2 = 1 - np.cos(t)
    y2 = -np.sin(t) + 0.5
    x2 += np.random.normal(0, noise, n1)
    y2 += np.random.normal(0, noise, n1)

    X = np.vstack([np.column_stack([x1, y1]),
                   np.column_stack([x2, y2])])
    y = np.array([0] * n1 + [1] * n1)

    return X, y


def gen_circles(n, noise):
    """
    Generate concentric circles dataset.

    Args:
        n (int): Total number of points.
        noise (float): Standard deviation of Gaussian noise.

    Returns:
        X (np.ndarray): 2D coordinates of points.
        y (np.ndarray): Labels (0 for inner circle, 1 for outer circle).
    """
    n1 = n // 2
    t1 = np.linspace(0, 2*np.pi, n1)
    t2 = np.linspace(0, 2*np.pi, n1)

    x1 = np.cos(t1) + np.random.normal(0, noise, n1)
    y1 = np.sin(t1) + np.random.normal(0, noise, n1)

    x2 = 2*np.cos(t2) + np.random.normal(0, noise, n1)
    y2 = 2*np.sin(t2) + np.random.normal(0, noise, n1)

    X = np.vstack([np.column_stack([x1, y1]),
                   np.column_stack([x2, y2])])

    y = np.array([0] * n1 + [1] * n1)
    return X, y


def gen_spiral(n, noise):
    """
    Generate a spiral dataset.

    Args:
        n (int): Total number of points.
        noise (float): Standard deviation of Gaussian noise.

    Returns:
        X (np.ndarray): 2D coordinates of points.
        y (np.ndarray): Labels (all zeros, as spiral is one cluster).
    """
    t = np.linspace(0, 4*np.pi, n)
    r = 0.1 * t
    x = r * np.cos(t) + np.random.normal(0, noise, n)
    y = r * np.sin(t) + np.random.normal(0, noise, n)
    return np.column_stack([x, y]), np.zeros(n, dtype=int)


def gen_mixed(points, noise):
    """
    Generate a mixed dataset combining Gaussian, moons, circles, and spiral.

    Args:
        points (int): Number of points per component.
        noise (float): Gaussian noise for non-linear structures.

    Returns:
        X (np.ndarray): 2D coordinates of all points.
        labels (np.ndarray): Labels for each point, unique per component.
    """
    # Gaussian clusters
    g, lg = gen_gaussian(points, [(0, 0), (4, 4), (-3, 4)], 0.5)

    # Shift moons
    m, lm = gen_two_moons(points, noise)
    m += np.array([8, 0])   # shift right

    # Shift circles
    c, lc = gen_circles(points, noise)
    c += np.array([-8, -2]) # shift left-down

    # Shift spiral
    s, ls = gen_spiral(points, noise)
    s += np.array([0, -8])  # shift down

    X = np.vstack([g, m, c, s])
    labels = np.concatenate([
        lg,
        lm + np.max(lg) + 1,
        lc + np.max(lg) + np.max(lm) + 2,
        ls + np.max(lg) + np.max(lm) + np.max(lc) + 3
    ])
    return X, labels


# ==========================================================
#   SAVE + PLOT
# ==========================================================

def save_dataset(name, X, labels):
    """
    Save dataset to CSV file.

    Args:
        name (str): Base filename (without folder).
        X (np.ndarray): 2D coordinates.
        labels (np.ndarray): Labels for each point.

    Saves:
        data/<name>.csv
    """
    os.makedirs("data", exist_ok=True)
    df = pd.DataFrame({"x": X[:, 0], "y": X[:, 1], "label": labels})
    df.to_csv(f"data/{name}.csv", index=False)
    print(f"Saved CSV → data/{name}.csv")


def plot_dataset(name, X, labels):
    """
    Plot dataset and save figure.

    Args:
        name (str): Base filename (used for plot name).
        X (np.ndarray): 2D coordinates.
        labels (np.ndarray): Labels for coloring points.

    Saves:
        plots/<name>.png
    """
    plt.figure(figsize=(6, 6))
    for lab in np.unique(labels):
        pts = X[labels == lab]
        plt.scatter(pts[:, 0], pts[:, 1], s=12, label=f"{lab}")

    plt.legend()
    plt.title(name)
    plt.axis("equal")

    os.makedirs("plots", exist_ok=True)
    out = f"plots/{name}.png"
    plt.savefig(out, dpi=200)
    print(f"Saved figure → {out}")

    plt.show()

# ==========================================================
#   MAIN ENTRY
# ==========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type",
        type=str,
        default="mixed",
        choices=["gaussian", "moons", "circles", "spiral", "mixed"],
        help="Dataset type (default: mixed)"
    )
    parser.add_argument("--points", type=int, default=300)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--name", type=str, default="dataset")
    
    args = parser.parse_args()

    # Auto-selection based on type
    if args.type == "gaussian":
        X, y = gen_gaussian(args.points,
                            [(0, 0), (4, 4), (-4, 4)],
                            0.5)
    elif args.type == "moons":
        X, y = gen_two_moons(args.points, args.noise)
    elif args.type == "circles":
        X, y = gen_circles(args.points, args.noise)
    elif args.type == "spiral":
        X, y = gen_spiral(args.points, args.noise)
    else:  # mixed by default
        X, y = gen_mixed(args.points, args.noise)

    save_dataset(args.name, X, y)
    plot_dataset(args.name, X, y)

