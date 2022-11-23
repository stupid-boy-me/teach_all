|评价指标分类模型评价指标|学习链接|
|-|-|
|**1.混淆矩阵**|[https://github.com/stupid-boy-me/teach_all/blob/main/pytorch_classifier/Confusion_Matrix/docs/1.分类模型评判指标 - 混淆矩阵.md](https://github.com/stupid-boy-me/teach_all/blob/main/pytorch_classifier/Confusion_Matrix/docs/1.分类模型评判指标 - 混淆矩阵.md)|
|**2.ROC曲线**|[https://github.com/stupid-boy-me/teach_all/blob/main/pytorch_classifier/Confusion_Matrix/docs/2.分类模型评判指标- ROC曲线与AUC面积.md](https://github.com/stupid-boy-me/teach_all/blob/main/pytorch_classifier/Confusion_Matrix/docs/2.分类模型评判指标- ROC曲线与AUC面积.md)|
|**3.AUC面积**|[https://github.com/stupid-boy-me/teach_all/blob/main/pytorch_classifier/Confusion_Matrix/docs/2.分类模型评判指标- ROC曲线与AUC面积.md](https://github.com/stupid-boy-me/teach_all/blob/main/pytorch_classifier/Confusion_Matrix/docs/2.分类模型评判指标- ROC曲线与AUC面积.md)|


- **1.混淆矩阵**

  - **一级指标(最底层的)**

    - TP:真正类：**样本的真实类别是正类，并且模型识别的结果也是正类**。
    - FN:假负类：样本的真实类别是正类，但是模型将其识别为负类。
    - FP:假正类：样本的真实类别是负类，但是模型将其识别为正类。
    - TN:真负类：**样本的真实类别是负类，并且模型将其识别为负类**。

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
