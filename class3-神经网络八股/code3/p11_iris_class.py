# 和Sequential方法相比，改动的地方用##数字##标注出
import tensorflow as tf
from keras.layers import Dense  ##1##
from keras import Model  ##2##
from sklearn import datasets
import numpy as np

x_train = datasets.load_iris().data
y_train = datasets.load_iris().target

np.random.seed(116)
np.random.shuffle(x_train)
np.random.seed(116)
np.random.shuffle(y_train)
tf.random.set_seed(116)

##3##
class IrisModel(Model):
    def __init__(self):
        super(IrisModel, self).__init__()
        self.d1 = Dense(3, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())   # 在__init__函数中定义了要在call函数中调用的具有三个神经元的全连接网络Dense

    def call(self, x):
        y = self.d1(x)    # 在call函数中调用self.d1实现了从输入x输出y的前向传播
        return y
##4##
model = IrisModel()   # 实例化

model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=500, validation_split=0.2, validation_freq=20)

model.summary()