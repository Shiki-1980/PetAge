import os

def find_extra_image(val_dir, val_txt):
    # 读取 val.txt 中的图片名
    with open(val_txt, 'r') as f:
        val_images = set(line.split()[0] for line in f)  # 读取第一列（图片名）
    
    # 获取 valset 目录下所有的图片文件名
    valset_images = set(f for f in os.listdir(val_dir) if f.endswith('.jpg') or f.endswith('.png'))
    
    # 找出在 valset 中存在但在 val.txt 中不存在的图片
    extra_images = valset_images - val_images
    
    if extra_images:
        print("找到多余的图片：")
        for img in extra_images:
            print(img)
    else:
        print("没有多余的图片")

# 设置路径
val_dir = './valset'  # valset 文件夹的路径
val_txt = './annotations/val.txt'  # val.txt 的路径

# 调用函数
find_extra_image(val_dir, val_txt)
