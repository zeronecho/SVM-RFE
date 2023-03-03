# Feature selection using SelectFromModel

# Model-based and sequential feature selection
# 比较了两种特征选择方法： SelectFromModel一种基于特征重要性，另 SequentialFeatureSelection一种依赖于贪婪方法。
# 加载可从 scikit-learn 中获得的糖尿病数据集。使用 RidgeCV估算器了解特征的重要性。具有最高绝对值的特征coef_被认为是最重要的。
# 根据系数选择最重要的两个特征。就是SelectFromModel 为了那个。SelectFromModel 接受一个threshold参数并将选择其重要性（由系数定义）高于此阈值的特征。
# 由于只想选择 2 个特征，因此将此阈值设置为略高于第三个最重要特征的系数。

# 另一种选择特征的方法是使用 SequentialFeatureSelector （SFS）。SFS 是一个贪婪的过程，在每次迭代中，根据交叉验证分数选择最好的新特征添加到选择的特征中。
# 也就是说，从 0 个特征开始，选择得分最高的最佳单个特征。重复该过程，直到达到所需数量的所选特征。
# 也可以反其道而行之（backward SFS），即从所有的特征开始，贪婪地选择一个特征一个一个地移除。

# Loading the data
from sklearn.datasets import load_diabetes

diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
print(diabetes.DESCR)

# Feature importance from coefficients
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import RidgeCV

ridge = RidgeCV(alphas=np.logspace(-6, 6, num=5)).fit(X, y)
importance = np.abs(ridge.coef_)
feature_names = np.array(diabetes.feature_names)
plt.bar(height=importance, x=feature_names)
plt.title("Feature importances via coefficients")
plt.show()

# Selecting features based on importance
from sklearn.feature_selection import SelectFromModel
from time import time

threshold = np.sort(importance)[-3] + 0.01

tic = time()
sfm = SelectFromModel(ridge, threshold=threshold).fit(X, y)
toc = time()
print(f"Features selected by SelectFromModel: {feature_names[sfm.get_support()]}")
print(f"Done in {toc - tic:.3f}s")

# Selecting features with Sequential Feature Selection
from sklearn.feature_selection import SequentialFeatureSelector

tic_fwd = time()
sfs_forward = SequentialFeatureSelector(
    ridge, n_features_to_select=2, direction="forward"
).fit(X, y)
toc_fwd = time()

tic_bwd = time()
sfs_backward = SequentialFeatureSelector(
    ridge, n_features_to_select=2, direction="backward"
).fit(X, y)
toc_bwd = time()

print(
    "Features selected by forward sequential selection: "
    f"{feature_names[sfs_forward.get_support()]}"
)
print(f"Done in {toc_fwd - tic_fwd:.3f}s")
print(
    "Features selected by backward sequential selection: "
    f"{feature_names[sfs_backward.get_support()]}"
)
print(f"Done in {toc_bwd - tic_bwd:.3f}s")