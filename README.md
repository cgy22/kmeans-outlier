# KMeans Outlier

<img src="https://dornhub.eu.org/resources/images/TonhTzeAI/TonhTzeAI-logo-v1-light.svg" alt="TonhTzeAI-logo-v1-light" width="400"><img src="https://dornhub.eu.org/resources/images/TonhTzeAI/TonhTzeAI-logo-v1-dark.svg" alt="TonhTzeAI-logo-v1-dark" width="400">

---

## ℹ️ About the Project

**KMeans Outlier** is a pure NumPy implementation of KMeans clustering for unsupervised outlier detection. Built from scratch without relying on scikit-learn, it uses the distance from a sample to its nearest cluster center as the anomaly score and applies statistical thresholds to detect outliers.

The project includes a full pipeline: synthetic data generation, model training, model persistence, and inference on new data. It provides benchmark datasets that demonstrate the algorithm's performance across spherical, high-dimensional, and imbalanced data scenarios.

---

### ✨ Features

| Category | Features |
| :--- | :--- |
| **Algorithm** | KMeans++ initialization, matrix-optimized distance computation, multiple restarts for stability |
| **Detection** | Distance-to-nearest-center as anomaly score, threshold via mean+k*std or quantile |
| **Datasets** | Four synthetic datasets: spherical baseline, 8D high-dimensional, imbalanced clusters, and unlabeled inference data |
| **Pipeline** | Train, save model parameters, reload, and predict on new data with consistent standardization |
| **CLI** | Command-line arguments for dataset selection, cluster count, threshold method, and model prefix |
| **Evaluation** | Precision, recall, and F1 score computed when labels are available |
| **Self-check** | Independent script demonstrating the algorithm's correctness against sklearn's inertia |

---

### 🎮 Usage

| Command / Action | Function |
| :--- | :--- |
| `python train.py` | Train on default data (`data/sample_data.csv`) and save model with prefix `model` |
| `python train.py --data data/high_dim_data.csv --prefix model_hd` | Train on high-dimensional data and save with custom prefix |
| `python train.py --no-label` | Train on unlabeled data without computing metrics |
| `python predict.py` | Predict on default new data (`data/new_data.csv`) using model prefix `model` |
| `python predict.py --data data/sample_data.csv --label --prefix model` | Predict on labeled data and compute metrics |
| `python make_sample_data.py` | Generate all four benchmark CSV files |

---

### 🛠️ Tech Stack

- **Language:** Python 3.8+
- **Core Library:** NumPy (algorithm implemented from scratch, no scikit-learn in the KMeans class)
- **Optional:** scikit-learn (only for comparison in the self-check script)
- **Data Format:** CSV with header row, last column as optional label (0 for normal, 1 for outlier)
- **Model Persistence:** NumPy `.npy` files for cluster centers, threshold, mean, and standard deviation

---

### 📊 Results

| Dataset | Size | Dimensions | Threshold | Detected | Precision | Recall | F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Spherical baseline | 315 | 2 | 1.3974 | 14 / 15 | 1.0 | 0.933 | 0.966 |
| High-dimensional | 315 | 8 | 4.6269 | 15 / 15 | 1.0 | 1.0 | 1.0 |
| Imbalanced clusters | 285 | 2 | 1.4542 | 9 / 15 | 1.0 | 0.6 | 0.75 |

---

### 🚀 Download & Play

> There is no pre-built binary. Run directly from source:

1. Clone this repository:
   ```bash
   git clone https://github.com/cgy22/kmeans-outlier.git
   cd kmeans-outlier
   ```

2. Install dependencies:
   ```bash
   pip install numpy
   ```

3. Generate sample data (optional, CSV files are already included):
   ```bash
   python make_sample_data.py
   ```

4. Train the model (this will generate 4 `.npy` model files in the root directory):
   ```bash
   python train.py
   ```
   *Expected output: It will print inertia for k=2~6, compute the threshold, show precision/recall/F1, and save the model.*

5. Predict on new data:
   ```bash
   python predict.py
   ```
   *Expected output: It will load the saved model, print the threshold, the number of detected anomalies, and the row indices with their anomaly scores.*

---

### 📁 Project Structure

```
kmeans-outlier/
├── data/
│   ├── sample_data.csv
│   ├── high_dim_data.csv
│   ├── imbalanced_data.csv
│   └── new_data.csv
├── scripts/
│   └── kmeans_outlier_selfcheck.py
├── kmeans_outlier.py
├── make_sample_data.py
├── predict.py
├── train.py
└── README.md
```

---

### ©️ Copyright

**Copyright © 2026 TonhTze AI. All Rights Reserved.**

---

### 📄 License

This project is licensed under the **MIT License**.

---

A toast with Prickly Pear juice!🍹  
TonhTze AI

---

# KMeans Outlier

<img src="https://dornhub.eu.org/resources/images/TonhTzeAI/TonhTzeAI-logo-v1-light.svg" alt="TonhTzeAI-logo-v1-light" width="400"><img src="https://dornhub.eu.org/resources/images/TonhTzeAI/TonhTzeAI-logo-v1-dark.svg" alt="TonhTzeAI-logo-v1-dark" width="400">

---

## ℹ️ 关于项目

**KMeans Outlier** 是一个完全基于 NumPy 手写实现的 KMeans 聚类异常检测项目。不依赖 scikit-learn，利用样本到最近簇中心的距离作为异常分数，并结合统计学阈值进行无监督异常值检测。

项目包含合成数据生成、模型训练、模型持久化以及对新数据推理的完整流程。内置多组基准数据，展示算法在球形、高维以及簇不均衡等不同场景下的表现。

---

### ✨ 功能特性

| 类别 | 功能 |
| :--- | :--- |
| 🧠 **核心算法** | KMeans++ 初始化、矩阵运算优化距离计算、多次初始化取最优结果 |
| 🎯 **检测机制** | 以到最近簇中心的距离为异常分数，支持均值加倍数标准差或分位数设定阈值 |
| 📊 **样例数据** | 四组合成数据集：球形基准、八维高维、簇不均衡、无标签推理数据 |
| ⚙️ **完整流程** | 训练、保存模型参数、重新加载，并保证推理与训练使用同一套标准化参数 |
| 🖥️ **命令行支持** | 支持通过命令行参数选择数据集、簇数、阈值方式及模型保存前缀 |
| 📝 **效果评估** | 若数据自带标签，自动计算并打印精确率、召回率与 F1 分数 |
| 🔍 **独立自检** | 提供独立脚本，展示算法原理并与 sklearn 的 inertia 进行对比验证 |

---

### 🎮 操作说明

| 命令 / 操作 | 功能 |
| :--- | :--- |
| `python train.py` | 使用默认数据（`data/sample_data.csv`）训练并保存前缀为 `model` 的模型 |
| `python train.py --data data/high_dim_data.csv --prefix model_hd` | 使用高维数据训练，并指定模型前缀 |
| `python train.py --no-label` | 无标签数据训练，跳过指标计算 |
| `python predict.py` | 使用默认前缀 `model` 对默认新数据（`data/new_data.csv`）进行推理 |
| `python predict.py --data data/sample_data.csv --label --prefix model` | 对带标签数据推理并输出评估指标 |
| `python make_sample_data.py` | 一键生成四组基准 CSV 数据文件 |

---

### 🛠️ 技术栈

- **编程语言：** Python 3.8+
- **核心依赖：** NumPy（KMeans 算法完全手写，未调用 sklearn）
- **可选依赖：** scikit-learn（仅自检脚本中用于对比验证）
- **数据格式：** CSV 带表头，最后一列为可选标签（0 表示正常，1 表示异常）
- **模型持久化：** 使用 NumPy `.npy` 文件保存簇中心、阈值、均值与标准差

---

### 📊 实验结果

| 数据集 | 规模 | 维度 | 阈值 | 检出数量 | 精确率 | 召回率 | F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 球形基准 | 315 | 2 | 1.3974 | 14 / 15 | 1.0 | 0.933 | 0.966 |
| 高维数据 | 315 | 8 | 4.6269 | 15 / 15 | 1.0 | 1.0 | 1.0 |
| 簇不均衡 | 285 | 2 | 1.4542 | 9 / 15 | 1.0 | 0.6 | 0.75 |

---

### 🚀 下载与运行

> 本项目无需下载可执行文件，直接从源码运行即可：

1. 克隆本仓库：
   ```bash
   git clone https://github.com/cgy22/kmeans-outlier.git
   cd kmeans-outlier
   ```

2. 安装依赖：
   ```bash
   pip install numpy
   ```

3. 生成样例数据（可选，仓库中已包含 CSV）：
   ```bash
   python make_sample_data.py
   ```

4. 训练模型（运行后会在根目录生成 4 个 `.npy` 模型参数文件）：
   ```bash
   python train.py
   ```
   *预期输出：会打印 k=2~6 的 inertia、计算出的阈值、精确率/召回率/F1，并提示模型已保存。*

5. 对新数据进行推理：
   ```bash
   python predict.py
   ```
   *预期输出：会加载模型，打印阈值、异常检出总数，以及被判为异常的行号与对应分数。*

---

### 📁 项目结构

```
kmeans-outlier/
├── data/
│   ├── sample_data.csv
│   ├── high_dim_data.csv
│   ├── imbalanced_data.csv
│   └── new_data.csv
├── scripts/
│   └── kmeans_outlier_selfcheck.py
├── kmeans_outlier.py
├── make_sample_data.py
├── predict.py
├── train.py
└── README.md
```

---

### ©️ 版权

**Copyright © 2026 TonhTze AI. All Rights Reserved.**

---

### 📄 许可证

本项目采用 **MIT 许可证**。

---

敬上一杯刺梨汁！🍹  
TonhTze AI
