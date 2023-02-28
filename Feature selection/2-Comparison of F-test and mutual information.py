# Comparison of F-test and mutual information
# F检验和互信息的比较


# 此示例说明了单变量 F 检验统计量和互信息之间的差异。
# 我们考虑均匀分布在 [0, 1] 上的 3 个特征 x_1、x_2、x_3，目标取决于它们如下：
# y = x_1 + sin(6 * pi * x_2) + 0.1 * N(0, 1)，即第三个特征完全无关。
# 下面的代码绘制了 y 对单个 x_i 的依赖关系以及单变量 F 检验统计和互信息的归一化值。
# 由于 F 检验仅捕获线性相关性，因此它将 x_1 评为最具辨别力的特征。另一方面，互信息可以捕获变量之间的任何类型的依赖关系，并将 x_2 评为最具辨别力的特征，这可能更符合我们对这个例子的直觉感知。这两种方法都正确地将 x_3 标记为不相关。

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_regression, mutual_info_regression

np.random.seed(0)
X = np.random.rand(1000, 3)
y = X[:, 0] + np.sin(6 * np.pi * X[:, 1]) + 0.1 * np.random.randn(1000)

f_test, _ = f_regression(X, y)
f_test /= np.max(f_test)

mi = mutual_info_regression(X, y)
mi /= np.max(mi)

plt.figure(figsize=(15, 5))
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.scatter(X[:, i], y, edgecolor="black", s=20)
    plt.xlabel("$x_{}$".format(i + 1), fontsize=14)
    if i == 0:
        plt.ylabel("$y$", fontsize=14)
    plt.title("F-test={:.2f}, MI={:.2f}".format(f_test[i], mi[i]), fontsize=16)
plt.show()