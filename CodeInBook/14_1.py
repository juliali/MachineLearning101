import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC # "Support vector classifier"

# 定义函数plot_svc_decision_function用于绘制分割超平面和其两侧的辅助超平面
def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # 创建网格用于评价模型
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # 绘制超平面
    ax.contour(X, Y, P, colors='k',
                   levels=[-1, 0, 1], alpha=0.5,
                   linestyles=['--', '-', '--'])

    # 标识出支持向量
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                       model.support_vectors_[:, 1],
                       s=300, linewidth=1,  edgecolors='blue', facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

# 用make_blobs生成样本数据
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=50, centers=2,
                      random_state=0, cluster_std=0.60)

# 将样本数据绘制在直角坐标中
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn');
plt.show()

# 用线性核函数的SVM来对样本进行分类
model = SVC(kernel='linear')
model.fit(X, y)

# 在直角坐标中绘制出分割超平面、辅助超平面和支持向量
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(model);
plt.show()