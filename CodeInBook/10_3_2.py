from sklearn.linear_model import LogisticRegression
import pandas as pd

# 导入数据集
data = pd.read_csv('quiz.csv', delimiter=',')
used_features = ["Last Score", "Hours Spent"]
X = data[used_features].values
scores = data["Score"].values
X_train = X[:11]
X_test = X[11:]

# 逻辑回归——二分类法
passed = []
for i in range(len(scores)):
    if(scores[i] >= 60):
        passed.append(1)
    else:
        passed.append(0)

y_train = passed[:11]
y_test = passed[11:]
classifier = LogisticRegression(C=1e5)
classifier.fit(X_train, y_train)
y_predict = classifier.predict(X_test)
print(y_predict)