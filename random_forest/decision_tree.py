import numpy as np

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth          # 最大深度
        self.min_samples_split = min_samples_split  # 最小分裂样本数
        self.feature_index = None           # 分裂特征索引
        self.threshold = None               # 分裂阈值
        self.left = None                    # 左子树
        self.right = None                   # 右子树
        self.label = None                   # 叶节点类别

    def gini(self, y):
        # 计算基尼不纯度
        classes, counts = np.unique(y, return_counts=True)
        prob = counts / len(y)
        return 1 - np.sum(prob ** 2)

    def best_split(self, X, y):
        # 寻找最佳特征和阈值
        m_samples, n_features = X.shape
        best_gini = float('inf')
        best_feature, best_threshold = None, None

        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                left_idx = X[:, feature_idx] < threshold
                if np.sum(left_idx) == 0 or np.sum(left_idx) == m_samples:
                    continue
                gini_left = self.gini(y[left_idx])
                gini_right = self.gini(y[~left_idx])
                total_gini = (gini_left * np.sum(left_idx) + gini_right * np.sum(~left_idx)) / m_samples
                if total_gini < best_gini:
                    best_gini = total_gini
                    best_feature = feature_idx
                    best_threshold = threshold
        return best_feature, best_threshold

    def fit(self, X, y, depth=0):
        # 递归构建决策树
        if (self.max_depth is not None and depth >= self.max_depth) or len(y) < self.min_samples_split:
            self.label = np.argmax(np.bincount(y))  # 叶节点取众数类别
            return self

        self.feature_index, self.threshold = self.best_split(X, y)
        if self.feature_index is None:
            self.label = np.argmax(np.bincount(y))
            return self

        left_idx = X[:, self.feature_index] < self.threshold
        self.left = DecisionTree(self.max_depth, self.min_samples_split).fit(X[left_idx], y[left_idx], depth+1)
        self.right = DecisionTree(self.max_depth, self.min_samples_split).fit(X[~left_idx], y[~left_idx], depth+1)
        return self

    def predict(self, X):
        # 单样本预测
        if self.label is not None:
            return self.label
        if X.ndim == 1:  # 单样本
            if X[self.feature_index] < self.threshold:
                return self.left.predict(X)
            else:
                return self.right.predict(X)
        else:  # 多样本
            return np.array([self.predict(x) for x in X])