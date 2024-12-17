这是数据集，以及一些处理的操作，包括去重，更名，平滑。

使用的数据集包括：

- 原始数据集：trainset，valset

- 品种数据集：breed，其下格式为

  --品种名

  ​	--*.jpg

  来自https://www.kaggle.com/competitions/dog-breed-identification/data

- yolo分割后的数据集：croppedTrainset、croppedValset

- 分年龄的数据集，两类：

  - 均值附近的数据集，MiddleTrain、MiddleTest(20到90月)
  - 其余年龄的数据集，OtherTrain、OtherTest(<20,>90)

