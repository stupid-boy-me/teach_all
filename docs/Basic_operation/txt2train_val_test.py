# 从总的数据集中（有规律）选取固定大小部分作为训练集验证集测试集的总和。如此处选取总数据集的前10000句作为需要使用的数据集。
# 两次使用train_test_split()将选取的固定大小的数据集，按照一定的比例，如8：1：1随机划分为训练集，验证集，测试集。
# 并分别将划分好的数据集进行写入到固定目录下

from sklearn.model_selection import train_test_split


def write_data(datapath, line_sen_list):
    '''
    datapath: 需要写入的文件地址
    line_sen_list: 需要写入的文件内容行列表
    '''
    with open(datapath, 'w', encoding='utf-8') as o:
        o.write(''.join(line_sen_list))
        o.close()


def main():
    raw_data_path = r'D:\nextvpu\yanye\yanye_xia\train_data_xia_deeplabv3\train_yanye_xia_B01\ImageSets\Segmentation\default.txt'
    train_data_path = r'D:\nextvpu\yanye\yanye_xia\train_data_xia_deeplabv3\train_yanye_xia_B01\ImageSets\Segmentation/train.txt'
    validate_data_path = r'D:\nextvpu\yanye\yanye_xia\train_data_xia_deeplabv3\train_yanye_xia_B01\ImageSets\Segmentation/val.txt'
    test_data_path = r'D:\nextvpu\yanye\yanye_xia\train_data_xia_deeplabv3\train_yanye_xia_B01\ImageSets\Segmentation/test.txt'

    line_sen_list = []

    with open(raw_data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        # 按某种规律选取固定大小的数据集
        for line in lines[0:582]:
            line_sen_list.append(''.join(line))
        f.close()

    label_list = [0] * 582  # 由于该数据形式为文本，且形式为数据和标签在一起，所以train_test_split()中标签可以给一个相同大小的0值列表，无影响。

    # 先将1.训练集，2.验证集+测试集，按照8：2进行随机划分
    X_train, X_validate_test, _, y_validate_test = train_test_split(line_sen_list, label_list, test_size=0.2,
                                                                    random_state=42)
    # 再将1.验证集，2.测试集，按照1：1进行随机划分
    X_validate, X_test, _, _ = train_test_split(X_validate_test, y_validate_test, test_size=0.5, random_state=42)

    # 分别将划分好的训练集，验证集，测试集写入到指定目录
    write_data(train_data_path, X_train)
    write_data(validate_data_path, X_validate)
    write_data(test_data_path, X_test)


if __name__ == '__main__':
    main()
