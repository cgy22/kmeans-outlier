import argparse
import numpy as np
from kmeans_outlier import read_data, standardize, load_model, evaluate


def main():
    parser = argparse.ArgumentParser(description="KMeans 异常检测推理脚本")
    parser.add_argument("--data", default="data/new_data.csv", help="新数据 CSV 路径")
    parser.add_argument("--label", action="store_true", help="新数据带标签，计算指标")
    parser.add_argument("--prefix", default="model", help="模型加载前缀，需与训练时一致")
    args = parser.parse_args()

    X_new, y_new = read_data(args.data, has_label=args.label)
    print("读取新数据", args.data, "规模", X_new.shape)

    model, threshold, mean, std = load_model(prefix=args.prefix)

    X_norm, _, _ = standardize(X_new, mean, std)

    score = model.distance_to_centers(X_norm).min(axis=1)
    is_outlier = score > threshold

    print("阈值", round(threshold, 4), "检出数量", int(is_outlier.sum()))
    for i, flag in enumerate(is_outlier):
        if flag:
            print("第", i, "行判为异常，分数", round(float(score[i]), 4))

    if y_new is not None:
        m = evaluate(y_new, is_outlier)
        print("精确率", round(m["precision"], 3),
              "召回率", round(m["recall"], 3),
              "F1", round(m["f1"], 3))


if __name__ == "__main__":
    main()
