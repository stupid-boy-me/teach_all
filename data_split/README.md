这里面会放一些处理数据的脚本工具


data_split.py 是将数据划分成train、val、test[可以自己设置比例]
    一开始的数据格式 图片：![data_split](https://user-images.githubusercontent.com/56495543/194759282-b02c3802-92bb-43e7-8dce-764718c722b4.png)
    数据最后的格式  图片：![data_split2](https://user-images.githubusercontent.com/56495543/194759355-bc5b861d-ae1c-4f35-adda-264489f8da50.png)
data_split2.py   是将数据划分成 train_images_path, train_images_label, val_images_path, val_images_label
MyDataset.py   是自己定义自己的数据集，将数据加载进内存
mian.py 的流程是 先将数据通过data_split2.py 划分成train_images_path, train_images_label, val_images_path, val_images_label，然后通过MyDataset.py加载数据进内存，再通过torch.utils.data.DataLoader 加载进模型
