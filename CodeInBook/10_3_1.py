from sklearn.linear_model import LinearRegression
import pandas as pd

# 导入数据集
data = pd.read_csv('quiz.csv', delimiter=',')
used_features = ["Last Score", "Hours Spent"]
X = data[used_features].values
scores = data["Score"].values
X_train = X[:11]
X_test = X[11:]

# 线性回归
y_train = scores[:11]
y_test = scores[11:]
regr = LinearRegression()
regr.fit(X_train, y_train)
y_predict = regr.predict(X_test)
print(y_predict)