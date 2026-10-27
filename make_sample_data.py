import numpy as np
import os


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_csv(path, X, y=None, feature_prefix="f"):
    n_features = X.shape[1]
    header = ",".join(["%s%d" % (feature_prefix, i + 1) for i in range(n_features)])
    if y is not None:
        header += ",label"
        data = np.c_[X, y]
    else:
        data = X
    np.savetxt(path, data, delimiter=",", header=header, comments="", fmt="%.6f")
    print("已生成", path, "规模", data.shape)


def make_spherical(seed=42, n_per_cluster=100, n_outlier=15):
    rng = np.random.RandomState(seed)
    centers = np.array([[0.0, 0.0], [9.0, 0.0], [4.5, 8.0]])
    normal = np.vstack([rng.normal(c, 0.9, size=(n_per_cluster, 2)) for c in centers])
    outliers = []
    while len(outliers) < n_outlier:
        p = rng.uniform(-10.0, 20.0, size=2)
        if np.min(np.linalg.norm(centers - p, axis=1)) > 5.0:
            outliers.append(p)
    outliers = np.array(outliers)
    X = np.vstack([normal, outliers])
    y = np.r_[np.zeros(len(normal), dtype=int), np.ones(len(outliers), dtype=int)]
    return X, y


def make_high_dim(seed=7, dim=8, n_per_cluster=100, n_outlier=15):
    rng = np.random.RandomState(seed)
    centers = rng.normal(0, 4, size=(3, dim))
    normal = np.vstack([rng.normal(c, 1.0, size=(n_per_cluster, dim)) for c in centers])
    outliers = rng.uniform(-18, 18, size=(n_outlier, dim))
    X = np.vstack([normal, outliers])
    y = np.r_[np.zeros(len(normal), dtype=int), np.ones(len(outliers), dtype=int)]
    return X, y


def make_imbalanced(seed=11):
    rng = np.random.RandomState(seed)
    sizes = [220, 35, 15]
    centers = np.array([[0.0, 0.0], [8.0, 0.0], [4.0, 7.0]])
    normal = np.vstack([rng.normal(c, 0.8, size=(n, 2)) for c, n in zip(centers, sizes)])
    outliers = rng.uniform(-10, 18, size=(15, 2))
    X = np.vstack([normal, outliers])
    y = np.r_[np.zeros(len(normal), dtype=int), np.ones(len(outliers), dtype=int)]
    return X, y


def make_new_unlabeled(seed=23, n_per_cluster=60, n_outlier=15):
    rng = np.random.RandomState(seed)
    centers = np.array([[0.0, 0.0], [9.0, 0.0], [4.5, 8.0]])
    normal = np.vstack([rng.normal(c, 0.9, size=(n_per_cluster, 2)) for c in centers])
    outliers = []
    while len(outliers) < n_outlier:
        p = rng.uniform(-10.0, 20.0, size=2)
        if np.min(np.linalg.norm(centers - p, axis=1)) > 5.0:
            outliers.append(p)
    outliers = np.array(outliers)
    X = np.vstack([normal, outliers])
    return X, None


if __name__ == "__main__":
    ensure_dir("data")

    X, y = make_spherical()
    save_csv("data/sample_data.csv", X, y)

    X, y = make_high_dim()
    save_csv("data/high_dim_data.csv", X, y)

    X, y = make_imbalanced()
    save_csv("data/imbalanced_data.csv", X, y)

    X, _ = make_new_unlabeled()
    save_csv("data/new_data.csv", X, None)
