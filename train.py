import argparse
import numpy as np
from kmeans_outlier import (KMeans, detect_outliers, evaluate,
                            read_data, standardize, save_model)


def choose_k(X, k_range):
    """打印不同簇数下的 inertia，辅助人工选择 n_clusters"""
    print("不同簇数下的 inertia")
    for k in k_range:
        m = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
        print("  k", k, "inertia", round(float(m.inertia_), 3))
    print()


def main():
    parser = argparse.ArgumentParser(description="KMeans 异常检测训练脚本")
    parser.add_argument("--data", default="data/sample_data.csv", help="训练数据 CSV 路径")
    parser.add_argument("--no-label", action="store_true", help="数据无标签，不计算指标")
    parser.add_argument("--n-clusters", type=int, default=3, help="簇数")
    parser.add_argument("--method", choices=["std", "quantile"], default="std", help="阈值方式")
    parser.add_argument("--k", type=float, default=2.0, help="std 方式下的倍数")
    parser.add_argument("--quantile", type=float, default=95.0, help="quantile 方式下的分位数")
    parser.add_argument("--prefix", default="model", help="模型保存前缀")
    args = parser.parse_args()

    X, y = read_data(args.data, has_label=not args.no_label)
    print("读取数据", args.data, "规模", X.shape)
    if y is not None:
        print("标签为异常的样本数量", int(y.sum()))
    print()

    X_norm, mean, std = standardize(X)

    choose_k(X_norm, range(2, 7))

    res = detect_outliers(X_norm, n_clusters=args.n_clusters, method=args.method,
                          k=args.k, quantile=args.quantile, random_state=0)
    print("选定配置 n_clusters", args.n_clusters, "method", args.method,
          "k", args.k, "quantile", args.quantile)
    print("阈值", round(float(res["threshold"]), 4),
          "检出数量", int(res["is_outlier"].sum()))

    if y is not None:
        m = evaluate(y, res["is_outlier"])
        print("精确率", round(m["precision"], 3),
              "召回率", round(m["recall"], 3),
              "F1", round(m["f1"], 3))

    save_model(res["model"], float(res["threshold"]), mean, std, prefix=args.prefix)
    print("模型已保存，前缀", args.prefix)


if __name__ == "__main__":
    main()
