import os

def read_train_txt(train_txt_path):
    """读取 annotations/train.txt 文件，返回字典 {图片名: 原始标签}"""
    original_labels = {}
    with open(train_txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            image_name, label = parts[0], int(parts[1])
            original_labels[image_name] = label
    return original_labels

def read_train_predictions(predictions_txt_path):
    """读取 TrainPredictions.txt 文件，返回字典 {图片名: 预测标签}"""
    predictions = {}
    with open(predictions_txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            image_name, prediction = parts[0], float(parts[1])  # 修改为浮动小数
            predictions[image_name] = round(prediction)  # 四舍五入为整数
    return predictions

def read_big_loss_directory(big_loss_dir_path):
    """读取 BigLoss 目录，返回包含大误差图片名的集合"""
    big_loss_images = set()
    for image_name in os.listdir(big_loss_dir_path):
        if image_name.endswith('.jpg') or image_name.endswith('.png'):
            big_loss_images.add(image_name)
    return big_loss_images

def generate_smoth_train_file(original_labels, predictions, big_loss_images, output_file):
    """生成新的 smothTrain.txt 文件，替换大误差图片的标签"""
    with open(output_file, 'w') as f:
        for image_name, original_label in original_labels.items():
            if image_name in big_loss_images:
                # 替换为训练预测的标签
                new_label = predictions.get(image_name, original_label)
                f.write(f"{image_name}\t{new_label}\n")
            else:
                # 保留原始标签
                f.write(f"{image_name}\t{original_label}\n")
def main():
    # 定义文件路径
    train_txt_path = 'annotations/train.txt'  # 原始标签路径
    predictions_txt_path = '../regression/prediction/regnetx_320/TrainPredictions.txt'  # 训练预测路径
    big_loss_dir_path = 'BigLoss'  # 大误差图片所在目录
    output_file = './annotations/smothTrain.txt'  # 输出的平滑后的训练文件

    # 读取数据
    original_labels = read_train_txt(train_txt_path)
    predictions = read_train_predictions(predictions_txt_path)
    big_loss_images = read_big_loss_directory(big_loss_dir_path)

    # 生成平滑后的训练文件
    generate_smoth_train_file(original_labels, predictions, big_loss_images, output_file)
    print(f"Smooth training file generated: {output_file}")

if __name__ == "__main__":
    main()
