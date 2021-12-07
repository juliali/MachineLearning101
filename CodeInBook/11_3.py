from sklearn import tree
from sklearn.model_selection import train_test_split
import numpy as np

# 9 个体验差和 8 个体验好的样本数据，对应 7 个特征，Yes 取值为 1，No 为 0
features = np.array([
[1, 1, 0, 0, 1, 0, 1],
[1, 1, 1, 0, 0, 0, 1],
[0, 1, 0, 0, 0, 0, 1],
[1, 1, 0, 0, 1, 0, 1],
[0, 1, 0, 0, 1, 0, 0],
[0, 1, 0, 0, 1, 0, 1],
[1, 1, 0, 0, 1, 0, 1],
[0, 1, 0, 0, 1, 0, 1],
[0, 1, 0, 1, 1, 1, 1],
[1, 0, 1, 1, 1, 1, 0],
[0, 0, 0, 1, 1, 1, 0],
[1, 0, 1, 1, 1, 1, 0],
[0, 0, 0, 1, 1, 1, 0],
[1, 0, 0, 1, 1, 1, 0],
[0, 0, 1, 0, 1, 1, 0],
[1, 1, 1, 1, 1, 1, 0],
[1, 0, 1, 1, 1, 1, 0]
])

# 0 表示是体验“差”，1 表示是体验“好”
labels = np.array([[0], [0], [0], [0], [0], [0], [0], [0], [0], [1],
[1], [1], [1], [1], [1], [1], [1]])

# 从数据集中取 20% 作为测试集，其他作为训练集
X_train, X_test, y_train, y_test = train_test_split(
features,
labels,
test_size=0.2,
random_state=0,
)

# 训练分类树模型
clf = tree.DecisionTreeClassifier()
clf.fit(X=X_train, y=y_train)

# 测试
print(clf.predict(X_test))

# 对比测试结果和预期结果
print(clf.score(X=X_test, y=y_test))

# 预测新餐馆
new_restaurant = np.array([[1, 1, 1, 1, 1, 1, 1]])
print(clf.predict(new_restaurant))