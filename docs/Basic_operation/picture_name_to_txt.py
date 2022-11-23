import cv2
import os
from tqdm import tqdm

def picture_name_to_txt(path,txt_path):
    filenames = [filename for filename in os.listdir(path)]

    for file_name in filenames:
        # if os.path.exists(txt_path):
        #     os.remove(txt_path)
        with open(txt_path,"a+") as f:
            print(file_name.split('.')[0],"1",file=f)

if __name__ == '__main__':
    root = r"D:\nextvpu\OCR\book_object\data\cornel_data\task1\yolo2xml\\"
    picture_path = os.path.join(root,"JPEGImages")
    txt_path = r"D:\nextvpu\OCR\book_object\data\cornel_data\task1\yolo2xml\\output.txt"

    picture_name_to_txt(picture_path,txt_path)
