import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from src.data.loaddata import load_data
from model import MLP

# 加载数据
subject_id = 1
base_path = "data"
all_X, all_y = load_data(subject_id, base_path)

# 合并数据
X = np.vstack(all_X).astype(np.float32)  # 转换为 float32 以兼容 PyTorch
y = np.concatenate(all_y).astype(np.int64)  # 目标变量转换为 int64

# SMOTE 过采样
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=0)

# 转换为 PyTorch 张量
tensor_X_train = torch.tensor(X_train)
tensor_y_train = torch.tensor(y_train)
tensor_X_test = torch.tensor(X_test)
tensor_y_test = torch.tensor(y_test)

# 数据加载器
train_dataset = TensorDataset(tensor_X_train, tensor_y_train)
test_dataset = TensorDataset(tensor_X_test, tensor_y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# 初始化模型
input_dim = X.shape[1]
output_dim = len(np.unique(y))
model = MLP(input_dim, hidden_dim=128, output_dim=output_dim)

# 选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

# 评估模型
model.eval()
y_pred_list = []
y_true_list = []
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X = batch_X.to(device)
        outputs = model(batch_X)
        _, predicted = torch.max(outputs, 1)
        y_pred_list.extend(predicted.cpu().numpy())
        y_true_list.extend(batch_y.numpy())

accuracy = accuracy_score(y_true_list, y_pred_list)
f1 = f1_score(y_true_list, y_pred_list, average='weighted')

print(f"Test Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
