# 1-import
import tensorflow as tf
from sklearn import datasets
import numpy as np

# 2-train test
'''
测试集的输入特征x_test 和 标签y_test 可以像x_train和y_train一样直接从数据集获取
也可以在fit中按比例从训练集中划分（本代码采用这种方式，所以只需加载x_train和y_train即可）
'''
x_train = datasets.load_iris().data
y_train = datasets.load_iris().target
# 以下代码实现了数据集的乱序
np.random.seed(116)
np.random.shuffle(x_train)
np.random.seed(116)
np.random.shuffle(y_train)
tf.random.set_seed(116)

# 3-models.Sequential   逐层搭建网络结构
'''
单层全连接神经网络，三个参数分别为：
神经元个数；网络所使用的激活函数；正则化方法
'''
model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(3, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())
])

# 4-model.compile   配置训练方法
model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1),   # SGD优化器，学习率设置为0.1
              # 由于神经网络输出使用了softmax激活函数，使得输出是概率分布，而不是原始输出，故from_logits=False
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              # iris数据集的标签是0/1/2这样的数值，而网络前向传播输出为概率分布
              metrics=['sparse_categorical_accuracy'])

# model.fit   执行训练过程
model.fit(x_train,   # 训练集输入特征
          y_train,   # 训练集标签
          batch_size=32,    # 训练时一次喂入神经网络多少组数据
          epochs=500,    # 数据集迭代循环多少次
          validation_split=0.2,    # 从训练集中选择20%的数据作为测试集
          validation_freq=20)   # 每迭代20次训练集要在测试集中验证一次准确率

# model.summary  打印网络结构，统计参数数目
model.summary()