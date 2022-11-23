## 分类模型评价指标

|评价指标分类模型评价指标|学习链接|
|-|-|
|**1.混淆矩阵**|[https://github.com/stupid-boy-me/teach_all/blob/main/pytorch_classifier/Confusion_Matrix/docs/1.Confusion_MAatrix.md](https://github.com/stupid-boy-me/teach_all/blob/main/pytorch_classifier/Confusion_Matrix/docs/1.Confusion_MAatrix.md)|
|**2.ROC曲线**|[https://github.com/stupid-boy-me/teach_all/blob/main/pytorch_classifier/Confusion_Matrix/docs/2.ROC_AOU.md](https://github.com/stupid-boy-me/teach_all/blob/main/pytorch_classifier/Confusion_Matrix/docs/2.ROC_AOU.md)|
|**3.AUC面积**|[https://github.com/stupid-boy-me/teach_all/blob/main/pytorch_classifier/Confusion_Matrix/docs/2.ROC_AOU.md](https://github.com/stupid-boy-me/teach_all/blob/main/pytorch_classifier/Confusion_Matrix/docs/2.ROC_AOU.md)|


- **1.混淆矩阵**

  - **一级指标(最底层的)**

    - TP

      真正类：**样本的真实类别是正类，并且模型识别的结果也是正类**。

    - FN

      假负类：样本的真实类别是正类，但是模型将其识别为负类。

    - FP

      假正类：样本的真实类别是负类，但是模型将其识别为正类。

    - TN

      真负类：**样本的真实类别是负类，并且模型将其识别为负类**。

  ### 从混淆矩阵得到分类指标

  - **二级指标**

    - 准确率（Accuracy）—— 针对整个模型

      

    - 精确率（Precision）

    - 灵敏度（Sensitivity）：就是召回率（Recall）= TPR

    - 特异度（Specificity）

      

    - P-R曲线

      

  - **三级指标**

    - **F1_score** 

    - **Fβ_Score**

- **2.ROC曲线和AUC面积**

- **3.实际应用：将[混淆矩阵，ROC曲线，AUC面积]用于MobileNetV3的应用案例**



[1.分类模型评判指标 - 混淆矩阵](https://www.wolai.com/88n2SjzMd1KpJyZL8bzebB)

[2.分类模型评判指标- ROC曲线与AUC面积](https://www.wolai.com/89brHhKdj5ZYNyXEe1kf2m)

参考链接：[https://blog.csdn.net/qq_44599368/article/details/121082272](https://blog.csdn.net/qq_44599368/article/details/121082272)

[https://zhuanlan.zhihu.com/p/265107997#:~:text=Python为输出的数据绘制表格 1 1．add_rows ()方法 2 2．draw ()方法 3,3．header ()方法 4 4．set_cols_align ()方法 5 5．set_cols_dtype ()方法](https://zhuanlan.zhihu.com/p/265107997#:~:text=Python为输出的数据绘制表格 1 1．add_rows ()方法 2 2．draw ()方法 3,3．header ()方法 4 4．set_cols_align ()方法 5 5．set_cols_dtype ()方法)

服务器的tensorboard如何可视化：[https://blog.csdn.net/qq_33431368/article/details/121943102](https://blog.csdn.net/qq_33431368/article/details/121943102)

[http://127.0.0.1:16006/](http://127.0.0.1:16006/)

# Pytorch 多分类模型绘制 ROC, PR 曲线

[https://blog.csdn.net/PanYHHH/article/details/110741286](https://blog.csdn.net/PanYHHH/article/details/110741286)

## 指定源安装

[https://blog.csdn.net/qq_43377653/article/details/127580666#:~:text=安装 sklearn 1. 安装 numpy scipy matplot pip3,只为当前用户 安装 ： pip3 install -- user scikit-learn](https://blog.csdn.net/qq_43377653/article/details/127580666#:~:text=安装 sklearn 1. 安装 numpy scipy matplot pip3,只为当前用户 安装 ： pip3 install -- user scikit-learn)

pip install scikit-learn -i [https://pypi.tuna.tsinghua.edu.cn/simple](https://pypi.tuna.tsinghua.edu.cn/simple)

