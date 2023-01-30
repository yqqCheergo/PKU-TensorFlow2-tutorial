import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense, \
    GlobalAveragePooling2D
from keras import Model

np.set_printoptions(threshold=np.inf)

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 由于Inception结构块中的卷积操作均采用了CBA结构：先卷积，再BN，再采用relu激活函数
# 故将其定义成一个新的类ConvBNRelu
class ConvBNRelu(Model):
    def __init__(self, ch, kernelsz=3, strides=1, padding='same'):   # ch表示特征图通道数，即卷积核个数；默认卷积核尺寸为3，步长是1，全零填充
        super(ConvBNRelu, self).__init__()
        self.model = tf.keras.models.Sequential([
            Conv2D(ch, kernelsz, strides=strides, padding=padding),
            BatchNormalization(),
            Activation('relu')
        ])

    def call(self, x):
        x = self.model(x, training=False)  # 在training=False时，BN通过整个训练集计算均值、方差去做批归一化；training=True时，通过当前batch的均值、方差去做批归一化。推理时 training=False效果好
        return x

# 构建InceptionNet基本单元，即Inception结构块
class InceptionBlk(Model):
    def __init__(self, ch, strides=1):
        super(InceptionBlk, self).__init__()
        self.ch = ch  # 通道数
        self.strides = strides  # 卷积步长
        self.c1 = ConvBNRelu(ch, kernelsz=1, strides=strides)
        self.c2_1 = ConvBNRelu(ch, kernelsz=1, strides=strides)
        self.c2_2 = ConvBNRelu(ch, kernelsz=3, strides=1)
        self.c3_1 = ConvBNRelu(ch, kernelsz=1, strides=strides)
        self.c3_2 = ConvBNRelu(ch, kernelsz=5, strides=1)
        self.p4_1 = MaxPool2D(3, strides=1, padding='same')
        self.c4_2 = ConvBNRelu(ch, kernelsz=1, strides=strides)

    def call(self, x):
        x1 = self.c1(x)
        x2_1 = self.c2_1(x)
        x2_2 = self.c2_2(x2_1)
        x3_1 = self.c3_1(x)
        x3_2 = self.c3_2(x3_1)
        x4_1 = self.p4_1(x)
        x4_2 = self.c4_2(x4_1)
        # concat along axis=channel
        x = tf.concat([x1, x2_2, x3_2, x4_2], axis=3)  # axis=3指定堆叠的维度是沿深度方向
        return x

# 精简版本的InceptionNet，共10层
class Inception10(Model):
    def __init__(self, num_blocks, num_classes, init_ch=16, **kwargs):   # 默认输出深度是16
        super(Inception10, self).__init__(**kwargs)
        self.in_channels = init_ch
        self.out_channels = init_ch
        self.num_blocks = num_blocks  # InceptionNet的block数
        self.init_ch = init_ch   # 初始通道数，也即InceptionNet基本单元的初始卷积核个数
        self.c1 = ConvBNRelu(init_ch)  # 16个3×3的卷积，步长为1，全零填充，BN，relu（在ConvBNRelu中都是默认的）
        # 四个Inception结构块顺序相连，每两个Inception结构块组成一个block
        self.blocks = tf.keras.models.Sequential()
        # 每经过一个block，特征图尺寸变为一半，通道数变为2倍
        for block_id in range(num_blocks):
            for layer_id in range(2):
                if layer_id == 0:
                    block = InceptionBlk(self.out_channels, strides=2)  # 每个block中的第一个Inception结构块步长是2，使得每个输出特征图尺寸减半
                else:
                    block = InceptionBlk(self.out_channels, strides=1)  # 每个block中的第二个Inception结构块步长是1
                self.blocks.add(block)
            # enlarger out_channels per block
            self.out_channels *= 2   # 把输出特征图深度加深，保证特征抽取中信息的承载量一致
        self.p1 = GlobalAveragePooling2D()   # 全局平均池化
        self.f1 = Dense(num_classes, activation='softmax')   # num_classes代表分类数，对于CIFAR10数据集即为10

    def call(self, x):
        x = self.c1(x)
        x = self.blocks(x)
        x = self.p1(x)
        y = self.f1(x)
        return y


model = Inception10(num_blocks=2, num_classes=10)  # 即block_0和block_1；十分类

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_save_path = "./checkpoint/Inception10.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)

history = model.fit(x_train, y_train, batch_size=1024, epochs=5, validation_data=(x_test, y_test), validation_freq=1,
                    callbacks=[cp_callback])
model.summary()

# print(model.trainable_variables)
file = open('./Inception10_weights.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

###############################################    show   ###############################################

# 显示训练集和验证集的acc和loss曲线
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()