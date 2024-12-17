import os
import shutil

def filter_and_copy_images(txt_file, source_dir, dest_dir, threshold=80.0):
    """
    读取txt文件，找出loss大于给定阈值的图片并复制到目标目录。

    :param txt_file: 存储图片和loss的txt文件路径
    :param source_dir: 图片文件的源目录
    :param dest_dir: 图片文件要复制到的目标目录
    :param threshold: loss的阈值，默认为80.0
    """
    # 确保目标目录存在
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    count =0
    # 读取txt文件
    with open(txt_file, 'r') as f:
        lines = f.readlines()

    # 遍历每一行，检查loss值
    for line in lines:
        # 拆分图片名和loss
        img_name, loss = line.strip().split('\t')
        loss = float(loss)

        # 如果loss大于阈值，则复制图片
        if loss > threshold:
            # 构造源图片路径
            source_path = os.path.join(source_dir, img_name)
            if os.path.exists(source_path):
                # 构造目标路径
                dest_path = os.path.join(dest_dir, img_name)
                shutil.copy(source_path, dest_path)
                count +=1
                print(f"复制图片: {img_name} 到 {dest_dir}")
            else:
                print(f"源文件不存在: {source_path}")
    print("共有",count,"张图片loss超过",threshold)

# 使用示例
if __name__ == "__main__":
    txt_file = '../regression/prediction/regnetx_320/train_loss.txt'  # 替换为txt文件的实际路径
    source_dir = '../DataSet/croppedTrainset'  # 替换为源图片目录路径
    dest_dir = '../DataSet/BigLoss'  # 替换为目标目录路径

    filter_and_copy_images(txt_file, source_dir, dest_dir, threshold=55.0)
