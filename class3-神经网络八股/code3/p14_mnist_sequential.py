# 用Sequential实现手写数字识别训练

# 1-import
import tensorflow as tf

# 2-train test
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0    # 对输入网络的输入特征进行归一化，使原本0-255之间的灰度值变成0-1之间的数值
# 把输入特征的数值变小更适合神经网络吸收

# 用Sequential搭建网络   3-models.Sequential
model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),   # 先把输入特征拉直为一维数组，即748个数值
        tf.keras.layers.Dense(128, activation='relu'),    # 定义第一层网络有128个神经元
        tf.keras.layers.Dense(10, activation='softmax')    # 定义第二层网络有10个神经元，使输出符合概率分布
])

# 用compile配置训练方法   4-model.compile
model.compile(optimizer='adam',   # 优化器
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),   # 损失函数
              # 由于第二层网络用了softmax让输出符合概率分布了，不是直接输出的，所以from_logits=False
              # 如果输出不满足概率分布，要=True
              metrics=['sparse_categorical_accuracy'])  # 数据集中的标签是数值，神经网络输出y是概率分布

# 在fit中执行训练过程   5-model.fit
model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_freq=1)   # validation_freq=1 每迭代一次训练集，执行一次测试集的评测

# 打印出网络结构和参数统计   6-model.summary
model.summary()