import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

# ==========================================================
#   DATA GENERATORS
# ==========================================================

def gen_gaussian(n, centers, spread):
    data = []
    labels = []
    for i, (cx, cy) in enumerate(centers):
        x = np.random.normal(cx, spread, n)
        y = np.random.normal(cy, spread, n)
        data.append(np.column_stack([x, y]))
        labels += [i] * n
    return np.vstack(data), np.array(labels)

def gen_two_moons(n, noise):
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
    n1 = n // 2
    t1 = np.linspace(0, 2*np.pi, n1)
    t2 = np.linspace(0, 2*np.pi, n1)

    x1 = np.cos(t1) + np.random.normal(0, noise, n1)
    y1 = np.sin(t1) + np.random.normal(0, noise, n1)

    x2 = 2.5*np.cos(t2) + np.random.normal(0, noise, n1)
    y2 = 2.5*np.sin(t2) + np.random.normal(0, noise, n1)

    X = np.vstack([np.column_stack([x1, y1]),
                   np.column_stack([x2, y2])])
    y = np.array([0] * n1 + [1] * n1)
    return X, y

def gen_spiral(n, noise):
    """ Generates two intertwined spirals """
    n_points = n // 2
    theta = np.sqrt(np.random.rand(n_points)) * 4 * np.pi

    # Spiral A
    r_a = 2 * theta + np.pi
    data_a = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a]).T
    x_a = data_a + np.random.randn(n_points, 2) * noise

    # Spiral B (Rotated)
    r_b = -2 * theta - np.pi
    data_b = np.array([np.cos(theta) * r_b, np.sin(theta) * r_b]).T
    x_b = data_b + np.random.randn(n_points, 2) * noise

    X = np.vstack([x_a, x_b])
    y = np.array([0] * n_points + [1] * n_points)
    return X, y

def gen_mixed(n_base, noise):
    """
    Args:
        n_base: Approximate points per component
    """
    # 1. Gaussian blobs (3 clusters)
    g, lg = gen_gaussian(n_base, [(0, 0), (4, 4), (-3, 4)], 0.5)

    # 2. Moons (2 clusters)
    m, lm = gen_two_moons(n_base, noise)
    m = m * 1.5 + np.array([8, 0])   # Scale & Shift

    # 3. Circles (2 clusters)
    c, lc = gen_circles(n_base, noise)
    c = c * 1.5 + np.array([-8, -2]) # Scale & Shift

    # 4. Spirals (2 clusters)
    s, ls = gen_spiral(n_base, noise)
    s = s / 3 + np.array([0, -8])    # Scale & Shift

    X = np.vstack([g, m, c, s])
    return X

def save_dataset(path, X):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(X)
    # Important: Save as x,y without index
    df.to_csv(path, header=["x", "y"], index=False)
    print(f"Dataset saved: {path} | Shape: {X.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Default 1250 per component * 6 logical groups = 7500 points
    parser.add_argument("--points", type=int, default=1250)
    parser.add_argument("--out", type=str, default="data/mixed_dataset.csv")

    args = parser.parse_args()

    print(f"Generating Mixed Dataset (Base={args.points})...")
    X = gen_mixed(args.points, 0.1)

    save_dataset(args.out, X)

    # Optional: Preview
    try:
        plt.figure(figsize=(10, 10))
        plt.scatter(X[:, 0], X[:, 1], s=1, alpha=0.6)
        plt.title(f"Dataset Preview: {X.shape[0]} points")
        preview_path = args.out.replace(".csv", ".png")
        plt.savefig(preview_path)
        print(f"Preview saved to {preview_path}")
    except:
        pass