import matplotlib.pyplot as plt
from collections import Counter,OrderedDict
import seaborn as sns
import pandas as pd

# 读取文件并统计月份分布
def analyze_pet_months(file_path):
    # 用于存储月份的计数
    month_counts = Counter()

    with open(file_path, 'r') as file:
        for line in file:
            # 去除空白字符并按制表符分割
            parts = line.strip().split('\t')
            if len(parts) == 2:
                try:
                    month = int(parts[1])
                    if month > 0:  # 确保是正整数
                        month_counts[month] += 1
                except ValueError:
                    print(f"跳过无效行: {line.strip()}")

    return month_counts

def clean_data(data, min_occurrences=10):
    """
    清洗数据，将计数小于min_occurrences的月份计数设置为0
    :param data: 字典，包含月份和计数
    :param min_occurrences: 最小出现次数阈值
    :return: 清洗后的数据字典
    """
    cleaned_data = {
        'Month': data['Month'],
        'Count': [count if count >= min_occurrences else 0 for count in data['Count']]
    }
    return cleaned_data


# 文件路径
file_path = './DataSet/annotations/train.txt'  # 替换为你的文件路径
file_name = file_path.split('/')[-1].rsplit('.', 1)[0]

# 统计并绘图
month_counts = analyze_pet_months(file_path)
# 将Counter转换为有序字典，确保月份按顺序排列
ordered_month_counts = OrderedDict(sorted(month_counts.items()))

# 确定横坐标的最大月份
max_month = max(ordered_month_counts.keys())

# 创建一个包含所有月份的数据框
months = list(range(1, max_month + 1))
data = {'Month': months, 'Count': [ordered_month_counts.get(month, 0) for month in months]}
cleaned_data = clean_data(data)
df = pd.DataFrame(cleaned_data)
plt.bar(data['Month'], data['Count'])

# 添加标题和轴标签
plt.title('Monthly Counts')
plt.xlabel('Month')
plt.ylabel('Count')

plt.savefig(file_name, dpi=300, bbox_inches='tight')
