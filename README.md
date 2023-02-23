# SVM-RFE
SVM-RFE的测试与备份

## 虚拟环境 (venv)
本项目创建一个 virtual environment，一个目录树，其中安装有特定Python版本，以及许多其他包。用于创建和管理虚拟环境的模块称为 venv。为了避免与其他软件包发生潜在冲突，强烈建议使用虚拟环境 (venv)

以pycharm为例，File – Settings – Project: pythonProject – Python Interpreters – Add。在终端可以看到前面出现了(venv)，表示现在在虚拟环境中运行；还可以通过Python Interpreter设置回到外部环境中。如果显示venv\Scripts\activate.ps1，因为在此系统上禁止运行脚本。`win+x` 以管理员方式运行powershell后，输入： `set-executionpolicy remotesigned`，之后重新打开Terminal即可解决。

官网参考：https://docs.python.org/3/tutorial/venv.html

## 安装 scikit-learn 环境

```
pip install -U scikit-learn
```
-u参数的含义是下载并更新到最新版。

```shell
python -m pip show scikit-learn  # to see which version and where scikit-learn is installed
python -m pip freeze  # to see all packages installed in the active virtualenv
python -c "import sklearn; sklearn.show_versions()"
```
检查安装。（python的-c 可以直接在命令行中调用python代码, 实际上-c 就是command 的意思。简言之就是python -c 可以在命令行中执行python 代码,）

官网参考：https://scikit-learn.org/stable/install.html 