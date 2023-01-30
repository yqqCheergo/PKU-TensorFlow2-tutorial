import tensorflow as tf
from matplotlib import pyplot as plt

# 导入数据集
mnist = tf.keras.datasets.mnist   # keras函数库提供了使用mnist数据集的接口
(x_train, y_train), (x_test, y_test) = mnist.load_data()   # load_data()直接从mnist中读取测试集和训练集

# 可视化训练集输入特征的第一个元素
plt.imshow(x_train[0], cmap='gray')  # 绘制灰度图
plt.show()

# 打印出训练集输入特征的第一个元素
print("x_train[0]:\n", x_train[0])    # 28行28列个像素值的二维数组（0表示纯黑色，255表示纯白色）
# 打印出训练集标签的第一个元素
print("y_train[0]:\n", y_train[0])    # 数值5

# 打印出整个训练集输入特征形状
print("x_train.shape:\n", x_train.shape)   # 6万个28行28列的数据
# 打印出整个训练集标签的形状
print("y_train.shape:\n", y_train.shape)   # 6万个标签
# 打印出整个测试集输入特征的形状
print("x_test.shape:\n", x_test.shape)   # 1万个28行28列的三维数据
# 打印出整个测试集标签的形状
print("y_test.shape:\n", y_test.shape)   # 1万个标签