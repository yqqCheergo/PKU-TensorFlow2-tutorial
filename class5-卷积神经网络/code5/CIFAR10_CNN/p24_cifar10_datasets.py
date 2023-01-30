import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

np.set_printoptions(threshold=np.inf)

# 导入CIFAR10数据集
cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 可视化训练集输入特征的第一个元素
plt.imshow(x_train[0])  # 绘制图片
plt.show()

# 打印出训练集输入特征的第一个元素
print("x_train[0]:\n", x_train[0])   # 是一个32行32列三通道的三维数组，是这张图片32行32列个像素点的RGB值
# 打印出训练集标签的第一个元素
print("y_train[0]:\n", y_train[0])   # [6]，对应frog（青蛙）

# 打印出整个训练集输入特征形状
print("x_train.shape:\n", x_train.shape)    # (50000, 32, 32, 3)
# 打印出整个训练集标签的形状
print("y_train.shape:\n", y_train.shape)    # (50000, 1)
# 打印出整个测试集输入特征的形状
print("x_test.shape:\n", x_test.shape)    # (10000, 32, 32, 3)  10000个32行32列的RGB三通道数据，维度是4
# 打印出整个测试集标签的形状
print("y_test.shape:\n", y_test.shape)    # (10000, 1)