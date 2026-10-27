import numpy as np


class KMeans:
    """使用 NumPy 自行实现的 KMeans 聚类算法"""

    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, n_init=10, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

    def _init_centers(self, X, rng):
        """kmeans++ 初始化，让初始中心彼此分散"""
        n_samples, n_features = X.shape
        centers = np.empty((self.n_clusters, n_features), dtype=np.float64)
        idx = rng.randint(n_samples)
        centers[0] = X[idx]
        closest_sq = np.sum((X - centers[0]) ** 2, axis=1)
        for i in range(1, self.n_clusters):
            total = closest_sq.sum()
            if total <= 0:
                probs = np.full(n_samples, 1.0 / n_samples)
            else:
                probs = closest_sq / total
            idx = rng.choice(n_samples, p=probs)
            centers[i] = X[idx]
            new_sq = np.sum((X - centers[i]) ** 2, axis=1)
            closest_sq = np.minimum(closest_sq, new_sq)
        return centers

    @staticmethod
    def _sq_distances(X, centers):
        """利用矩阵运算计算样本到各中心的平方距离，避免三维广播占内存"""
        x_sq = np.sum(X ** 2, axis=1)[:, None]
        c_sq = np.sum(centers ** 2, axis=1)[None, :]
        dist_sq = x_sq + c_sq - 2.0 * X.dot(centers.T)
        np.maximum(dist_sq, 0.0, out=dist_sq)
        return dist_sq

    def _fit_once(self, X, rng):
        """一次完整的迭代过程，返回中心、标签和簇内平方和"""
        centers = self._init_centers(X, rng)
        prev_inertia = np.inf
        for _ in range(self.max_iter):
            dist_sq = self._sq_distances(X, centers)
            labels = np.argmin(dist_sq, axis=1)
            inertia = dist_sq[np.arange(X.shape[0]), labels].sum()
            new_centers = np.empty_like(centers)
            for k in range(self.n_clusters):
                mask = labels == k
                if mask.any():
                    new_centers[k] = X[mask].mean(axis=0)
                else:
                    new_centers[k] = X[rng.randint(X.shape[0])]
            shift = np.sum((new_centers - centers) ** 2)
            centers = new_centers
            if abs(prev_inertia - inertia) < self.tol or shift < self.tol:
                break
            prev_inertia = inertia
        dist_sq = self._sq_distances(X, centers)
        labels = np.argmin(dist_sq, axis=1)
        inertia = dist_sq[np.arange(X.shape[0]), labels].sum()
        return centers, labels, inertia

    def fit(self, X):
        """多次初始化取 inertia 最小的一组结果，降低陷入局部最优的概率"""
        X = np.asarray(X, dtype=np.float64)
        rng = np.random.RandomState(self.random_state)
        best = None
        for _ in range(self.n_init):
            result = self._fit_once(X, rng)
            if best is None or result[2] < best[2]:
                best = result
        self.cluster_centers_, self.labels_, self.inertia_ = best
        return self

    def distance_to_centers(self, X):
        """返回每个样本到每个簇中心的欧氏距离矩阵"""
        X = np.asarray(X, dtype=np.float64)
        return np.sqrt(self._sq_distances(X, self.cluster_centers_))


def detect_outliers(X, n_clusters=3, method="std", k=3.0, quantile=95.0, random_state=0):
    """基于 KMeans 的异常检测

    method 为 std 时，阈值等于距离均值加 k 倍标准差
    method 为 quantile 时，阈值等于距离的指定分位数
    """
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state).fit(X)
    dist = model.distance_to_centers(X)
    score = dist.min(axis=1)
    if method == "std":
        threshold = score.mean() + k * score.std()
    elif method == "quantile":
        threshold = np.percentile(score, quantile)
    else:
        raise ValueError("method 只支持 std 或 quantile")
    is_outlier = score > threshold
    return {
        "model": model,
        "score": score,
        "threshold": threshold,
        "is_outlier": is_outlier,
    }


def evaluate(y_true, y_pred):
    """计算精确率、召回率和 F1"""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1}


def read_data(path, has_label=True):
    """从 CSV 读取数据，首行为表头，has_label 为真时最后一列是标签

    返回 X 和 y，无标签时 y 为 None
    """
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if has_label:
        X = data[:, :-1]
        y = data[:, -1].astype(int)
        return X, y
    return data, None


def standardize(X, mean=None, std=None):
    """标准化，训练时传 None 会计算参数并返回，推理时传入训练阶段的参数

    返回标准化后的数据以及均值和标准差
    """
    X = np.asarray(X, dtype=np.float64)
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return (X - mean) / std, mean, std


def save_model(model, threshold, mean, std, prefix="model"):
    """把簇中心、阈值、标准化参数保存为 npy 文件"""
    np.save(prefix + "_centers.npy", model.cluster_centers_)
    np.save(prefix + "_threshold.npy", np.array([threshold]))
    np.save(prefix + "_mean.npy", np.asarray(mean))
    np.save(prefix + "_std.npy", np.asarray(std))


def load_model(prefix="model"):
    """加载簇中心、阈值和标准化参数，返回一个只含中心的 KMeans 对象"""
    centers = np.load(prefix + "_centers.npy")
    threshold = float(np.load(prefix + "_threshold.npy")[0])
    mean = np.load(prefix + "_mean.npy")
    std = np.load(prefix + "_std.npy")
    model = KMeans(n_clusters=len(centers))
    model.cluster_centers_ = centers
    return model, threshold, mean, std
