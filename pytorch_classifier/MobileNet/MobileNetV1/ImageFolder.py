from torchvision import transforms as T
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
normalize = T.Normalize(mean=[0.5106151, 0.49528646, 0.40684184], std=[0.22255649, 0.22126609, 0.22495168])
transform  = T.Compose([
         T.RandomResizedCrop(224),
         T.RandomHorizontalFlip(),
         T.ToTensor(),
         normalize,
])
dataset = ImageFolder('/algdata01/yiguo.huang/hyg_deep_learning_me/Dataset/animals/', transform=transform)

# 深度学习中图片数据一般保存成CxHxW，即通道数x图片高x图片宽
#print(dataset[0][0].size())

to_img = T.ToPILImage()
# 0.2和0.4是标准差和均值的近似
a=to_img(dataset[0][0]*0.2+0.4)
plt.imshow(a)
plt.axis('off')
plt.show()
