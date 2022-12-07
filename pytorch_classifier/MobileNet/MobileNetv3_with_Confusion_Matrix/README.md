(1)MobileNetV3_large:适合硬件条件较好的设备

(2)MobileNetV3_small:适合硬件条件局限性较大的设备，精度略有下降

3.pytorch实现

'''

参考链接: [https://blog.csdn.net/baidu_38406307/article/details/107374989](https://blog.csdn.net/baidu_38406307/article/details/107374989)

完整项目地址: [https://gitcode.net/mirrors/yichaojie/mobilenetv3?utm_source=csdn_github_accelerator](https://gitcode.net/mirrors/yichaojie/mobilenetv3?utm_source=csdn_github_accelerator)

'''

1.MobileNetv3与MobileNetv1和MobileNetv2的比较

  MobileNet是Google公司推出的轻量化系列网络，用以在移动平台上进行神经网络的部署和应用。2019年，Google发布了第三代MobileNet，即MobileNetV3。

  在MobileNet系列的精度和计算量上都达到了新的state-of-art，以下简单回顾一下三代MobileNet的主要特性：

  (1)MobileNetV1:MobileNetV1的主要思想是将普通的卷积操作分解为两步，先做一次仅卷积，不求和的“深度卷积”操作，再使用1*1的“点卷积”对深度卷积

    得到的多通道结果进行融合，减少了大量的通道融合时间和参数。

  (2)MobileNetV2:ResNet中的Bottleneck残差块走下来是通道数先变少在变多，每层中的激活函数会导致丢失一部分的信息（神经元抑制），

    那么在输入通道数很多时丢的信息会很多，因此MobileNetV2把Bottleneck改为先将通道数增加，再将通道数减少，引入到MobileNetV2中，称之为“反转残差块”。

  (3)MobileNetV3：MobileNetV3主要的贡献就是使用"科学的理论"重新设计了网络结构，并且引入了H-Swish激活函数与Relu搭配使用。

    另外在网络中还引入了Squeeze-And-Excite模块，这个虽然在文中只是简单的一提，但是却很重要。复现的时候没有这个是达不到论文的效果的。

3.pytorch实现

  /algdata01/yiguo.huang/hyg_deep_learning_me/MobileNet/MobileNetv3/Mobilenetv3.py

## 这是将MobileNetv3**和**Confusion_Matrix(混淆矩阵)的结合应用

|MobileNetv3的视频地址：|[# lesson22_MobilenetV3的网络结构讲解](https://www.bilibili.com/video/BV1kR4y1f7vn/?spm_id_from=333.999.0.0&vd_source=5ba1bf3a19888ef725acbeaf5d3fc6e6)|
|-|-|
|MobileNetv3的代码地址：|[https://github.com/stupid-boy-me/teach_all/tree/main/pytorch_classifier/MobileNet/MobileNetV3](https://github.com/stupid-boy-me/teach_all/tree/main/pytorch_classifier/MobileNet/MobileNetV3)|
|MobileNetv3_with_Confusion_Matrix的视频地址：|[# lesson24_混淆矩阵和ROC在MobileNetV3的实际应用](https://www.bilibili.com/video/BV1724y1C7f1/?spm_id_from=333.999.0.0&vd_source=5ba1bf3a19888ef725acbeaf5d3fc6e6)|
|Confusion_Matrix的代码地址|[https://github.com/stupid-boy-me/teach_all/tree/main/pytorch_classifier/MobileNet/MobileNetv3_with_Confusion_Matrix](https://github.com/stupid-boy-me/teach_all/tree/main/pytorch_classifier/MobileNet/MobileNetv3_with_Confusion_Matrix)|


### 一：文件结构讲解

|文件名|结构功能|
|-|-|
|acquire_mean_std.py|计算分类数据集的均值和方差|
|dataset.py|自定义数据集|
|metrics.py   |混淆矩阵指标代码|
|Mobilenetv3.py|Mobilenetv3的代码|
|test.py|测试每一个类别准确率代码|
|test_with_PR.py|测试数据集，输出PR|
|test_with_ROC.py|测试数据集，输出ROC|
|train.py|训练脚本|
|transform_processing.py|transform|
|utils.py|常见脚本|


test_with_PR.py输出的PR图

![](https://secure2.wostatic.cn/static/bLEAJyoGi7L1ynjsWGnoLR/pr_curve.jpg?auth_key=1670417257-roZ6536ZqE69XJY7iL77Z8-0-1ce665d32b627aa1368f16248762fa12)

test_with_ROC.py输出的ROC图，同时能获取每个类别的AOU

![](https://secure2.wostatic.cn/static/iDGLqtfrWUpyPmRVPzqWsY/set113_roc.jpg?auth_key=1670417330-f3b3A11aC1yqTxBxr9XYzw-0-b075dbf469ef82bdfd22c8eeb8cccd91)

train.py输出的混淆矩阵图：

![](https://secure2.wostatic.cn/static/jKQgB99gBj8jXSbubnuimn/epoch-17.png?auth_key=1670417416-8xEC89zMQMobLXrhyStFW6-0-e2c70a3576803a407312eeb953566dee)

### 二：MobileNetV3_large网络结构图.jpg

![](https://secure2.wostatic.cn/static/qXTsFdL6axWMzhL6LEKxk5/MobileNetV3_large网络结构图.jpg?auth_key=1670417717-gWdTMeiZCRAy8fUaUfW8et-0-08a73a3b3d7f8b9f0ebdc42f5c37f4e5)

1. 第一列Input代表mobilenetV3每个特征层的shape变化；
2. 第二列Operator代表每次特征层即将经历的block结构，在MobileNetV3中，特征提取经过了许多的bneck结构，NBN为不适用BN层；
3. 第三、四列分别代表了bneck内倒残差结构升维后的通道数、输入到bneck时特征层的通道数。
4. 第五列SE代表了是否在这一层引入注意力机制。
5. 第六列NL代表了激活函数的种类，HS代表h-swish，RE代表RELU。
6. 第七列s代表了每一次block结构所用的步长。

### 三：MobileNetV3_small网络结构图.jpg

![](https://secure2.wostatic.cn/static/4ScPdFVnPa1RbxSFQUCsjo/MobileNetV3_small网络结构图.jpg?auth_key=1670417765-bWkqD43CzCsQfeUXC3q8oj-0-99431cf7fc16e68934c0b942ec17ef14)

### 四：分类数据类别

{

  "0": "butterfly",

  "1": "cat",

  "2": "dog",

  "3": "elephant",

  "4": "ragno"

}
