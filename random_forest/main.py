import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from random_forest import RandomForest
from ucimlrepo import fetch_ucirepo

print("Starting program...")

# 加载鸢尾花数据集
iris = fetch_ucirepo(id=53)
X = iris.data.features.values
y = iris.data.targets.values.squeeze()
y_labels, y = np.unique(y, return_inverse=True)  # 标签编码为整数

print("Data loaded successfully.")

# 检查数据加载
print("X shape:", X.shape)
print("y shape:", y.shape)
print("X sample:", X[:5])
print("y sample:", y[:5])

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data split successfully.")

# 检查数据划分
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# 训练随机森林
print("Training random forest...")
rf = RandomForest(n_trees=100, max_depth=5)
rf.fit(X_train, y_train)

print("Random forest trained successfully.")

# 检查训练结果
print("Number of trees trained:", len(rf.trees))

# 预测并评估准确率
print("Making predictions...")
y_pred = rf.predict(X_test)
print("Predictions:", y_pred)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")

# 特征重要性计算（基于分裂次数）
print("Calculating feature importances...")
feature_importances = np.zeros(X.shape[1])
for tree in rf.trees:
    if tree.feature_index is not None:
        feature_importances[tree.feature_index] += 1
feature_importances /= len(rf.trees)

# 可视化
print("Plotting feature importance...")
feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
plt.bar(feature_names, feature_importances)
plt.title('Feature Importance')
plt.savefig('../feature_importance.png')
plt.show()

print("Program completed successfully.")