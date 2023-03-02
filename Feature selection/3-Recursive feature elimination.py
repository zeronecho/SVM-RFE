# Recursive feature elimination
# 递归特征消除

# 给定一个为特征分配权重的外部估计器（例如，线性模型的系数），递归特征消除 ( ) 的目标RFE是通过递归地考虑越来越小的特征集来选择特征。
# 首先，估计器在初始特征集上进行训练，每个特征的重要性通过任何特定属性（例如coef_，feature_importances_）或可调用获得。
# 然后，从当前特征集中删除最不重要的特征。该过程在修剪后的集合上递归重复，直到最终达到所需的特征选择数量。

from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt

# Load the digits dataset
digits = load_digits()
X = digits.images.reshape((len(digits.images), -1))
y = digits.target

# Create the RFE object and rank each pixel
svc = SVC(kernel="linear", C=1)
rfe = RFE(estimator=svc, n_features_to_select=1, step=1)
rfe.fit(X, y)
ranking = rfe.ranking_.reshape(digits.images[0].shape)

# Plot pixel ranking
plt.matshow(ranking, cmap=plt.cm.Blues)
plt.colorbar()
plt.title("Ranking of pixels with RFE")
plt.show()