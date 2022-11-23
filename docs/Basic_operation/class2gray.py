import cv2
import os
from tqdm import tqdm
path = r"D:\nextvpu\yanye\analyse_yangeng\data\data_biaozhu\task_1\SegmentationClass_new\\"
out_path = r"D:\nextvpu\yanye\analyse_yangeng\data\data_biaozhu\task_1\SegmentationClass_0_255\\"
filenames = [filename for filename in os.listdir(path)]
for name in tqdm(filenames):
    filename_path = os.path.join(path,name)
    img = cv2.imread(filename_path)
    imGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(imGray.shape)
    h,w = imGray.shape[0],imGray.shape[1]
    for i in range(h-1):
        for j in range(w-1):
            if imGray[i][j] > 0 :
                print(imGray[i][j])
                imGray[i][j] = 1
                print(imGray[i][j])
                print('#########################')
    out_path_filename = os.path.join(out_path,name)
    cv2.imwrite(out_path_filename,imGray)
