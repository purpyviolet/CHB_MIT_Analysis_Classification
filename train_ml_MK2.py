import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from src.data.loaddata import load_data
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

# 创建一个保存结果的文件夹
subject_id = 1
base_path = "data"
results_dir = f"results_ml/{subject_id}"
os.makedirs(results_dir, exist_ok=True)

# 检查是否存在特征文件
features_file = f"features/features_{subject_id}.npy"

# 如果存在特征文件，则直接加载；否则，提取并保存特征
if os.path.exists(features_file):
    print("Loading pre-extracted features...")
    data = np.load(features_file)
    X = data[:, :-1]  # 特征
    y = data[:, -1]   # 标签
else:
    print("Extracting features...")
    all_X, all_y = load_data(subject_id, base_path)

    # 合并 all_X 和 all_y
    X = np.vstack(all_X)
    y = np.concatenate(all_y)

    # 保存提取的特征
    data = np.column_stack((X, y))
    np.save(features_file, data)

# 初始化 SMOTE 实例
smote = SMOTE()

# 应用 SMOTE 过采样
X_resampled, y_resampled = smote.fit_resample(X, y)

# 分割处理后的数据集
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=0)

# 定义模型字典
models = {
    "XGBoost": xgb.XGBClassifier(random_state=0, use_label_encoder=False, eval_metric='mlogloss'),
    "LogisticRegression": LogisticRegression(random_state=0, max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "NaiveBayes": GaussianNB(),
    "LinearSVM": LinearSVC(random_state=0),
    # "MLP": MLPClassifier(hidden_layer_sizes=(100,), random_state=0, max_iter=100)
}

# 创建结果字典
results = {}

# 对每个模型进行训练、预测并计算评估指标
for model_name, model in tqdm(models.items(), desc="Training models", unit="model"):
    print(f"Training {model_name}...")

    # 训练模型
    model.fit(X_train, y_train)

    # 对测试集进行预测
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)

    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    # 保存每个模型的结果
    results[model_name] = {
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc
    }

    # 将结果保存到txt文件
    result_file = os.path.join(results_dir, f"{model_name}_results.txt")
    with open(result_file, "w") as f:
        f.write(f"{model_name} Results:\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"ROC AUC: {roc_auc:.4f}\n")

# 打印所有模型的结果
for model_name, metrics in results.items():
    print(f"\n{model_name} Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

# 可视化模型性能
metrics = ['accuracy', 'f1_score', 'precision', 'recall', 'roc_auc']
model_names = list(results.keys())

# 绘制每个模型的评估指标
fig, axes = plt.subplots(1, len(metrics), figsize=(20, 6))
for i, metric in enumerate(metrics):
    ax = axes[i]
    values = [results[model][metric] for model in model_names]
    ax.bar(model_names, values, color='skyblue')
    ax.set_title(f'{metric} Comparison')
    ax.set_ylabel(metric)
    ax.set_xticklabels(model_names, rotation=45)

plt.tight_layout()
plt.show()