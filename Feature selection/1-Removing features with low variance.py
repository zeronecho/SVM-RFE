# 1-Removing features with low variance.py 移除低方差的特征
# VarianceThreshold是一种简单的特征选择基线方法。它删除所有方差不满足某个阈值的特征。默认情况下，它会删除所有零方差特征，即在所有样本中具有相同值的特征。

# 假设有一个具有布尔特征的数据集，并且想要删除超过 80% 的样本中所有为 1 或 0（开或关）的特征。布尔特征是伯努利随机变量，这些变量的方差由Var[X]=p(1-p)给出,选择使用阈值：.8 * (1 - .8)

from sklearn.feature_selection import VarianceThreshold
X = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
sel.fit_transform(X)

print(sel.fit_transform(X))

# 运行此程序，结果显示VarianceThreshold已经删除了第一列，这有一个概率 p=5/6>.8包含一个零。
