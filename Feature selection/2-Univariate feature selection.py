# Univariate feature selection
# 单变量特征选择


# 单变量特征选择通过基于单变量统计测试选择最佳特征来工作。它可以看作是估计器的预处理步骤。Scikit-learn 将特征选择例程公开为实现该方法的对象transform：
#
# SelectKBest删除除k得分最高的特征
# SelectPercentile删除除用户指定的最高得分百分比之外的所有功能
# 对每个特征使用常见的单变量统计测试： false positive rate SelectFpr, false discovery rate SelectFdr, or family wise error SelectFwe.
#
# GenericUnivariateSelect允许使用可配置策略执行单变量特征选择。这允许使用超参数搜索估计器选择最佳的单变量选择策略。
#
# 例如，我们可以使用 F 检验来检索数据集的两个最佳特征，如下所示：

from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
X, y = load_iris(return_X_y=True)
print(X.shape)

X_new = SelectKBest(f_classif, k=2).fit_transform(X, y)
print(X_new.shape)


# 这些对象将一个评分函数作为输入，该函数返回单变量分数和 p 值（或仅 和 的分数SelectKBest） SelectPercentile：
#
# 对于回归：r_regression, f_regression,mutual_info_regression
# 对于分类：chi2, f_classif,mutual_info_classif
#
# 基于 F 检验的方法估计两个随机变量之间的线性相关程度。另一方面，互信息方法可以捕获任何类型的统计相关性，但由于是非参数的，因此需要更多样本才能进行准确估计。请注意，X^2-test 应仅应用于非负面特征，例如频率。
#
# 稀疏数据的特征选择
# 如果您使用稀疏数据（即表示为稀疏矩阵的数据）， chi2, mutual_info_regression,mutual_info_classif 将处理数据而不使其变得密集。