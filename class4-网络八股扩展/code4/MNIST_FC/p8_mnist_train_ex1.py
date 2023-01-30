import tensorflow as tf
# PIL是Python中常用的图像处理库，提供了诸如图像打开、缩放、旋转、颜色转换等常用功能
from PIL import Image   # 从PIL(Python Imaging Library)模块中导入Image类
import numpy as np
import os

train_path = './mnist_image_label/mnist_train_jpg_60000/'     # 训练集图片路径
train_txt = './mnist_image_label/mnist_train_jpg_60000.txt'   # 训练集标签文件
# 在使用训练好的模型时，有一种保存模型的文件格式叫.npy，是numpy专用的二进制文件
x_train_savepath = './mnist_image_label/mnist_x_train.npy'    # 训练集输入特征存储文件
y_train_savepath = './mnist_image_label/mnist_y_train.npy'    # 训练集标签存储文件

test_path = './mnist_image_label/mnist_test_jpg_10000/'     # 测试集图片路径
test_txt = './mnist_image_label/mnist_test_jpg_10000.txt'   # 测试集标签文件
x_test_savepath = './mnist_image_label/mnist_x_test.npy'    # 测试集输入特征存储文件
y_test_savepath = './mnist_image_label/mnist_y_test.npy'    # 测试集标签存储文件

def generateds(path, txt):    # path为图片路径，txt为标签文件
    f = open(txt, 'r')  # 以只读形式打开txt文件
    contents = f.readlines()  # 读取文件中所有行
    f.close()  # 关闭txt文件
    x, y_ = [], []  # 建立空列表
    for content in contents:  # 逐行取出
        value = content.split()  # 以空格分开，图片路径为value[0] , 标签为value[1] , 存入列表
        img_path = path + value[0]  # 拼出图片路径和文件名，为图片的索引路径
        img = Image.open(img_path)  # 读入图片
        # image = image.convert()是图像实例对象的一个方法，接受一个mode参数，用以指定一种色彩模式
        img = np.array(img.convert('L'))  # 图片变为8位宽度的灰度值，np.array格式
        img = img / 255.  # 数据归一化 （实现预处理）
        x.append(img)  # 归一化后的数据，贴到列表x
        y_.append(value[1])  # 标签贴到列表y_
        print('loading : ' + content)  # 打印状态提示
    x = np.array(x)  # 变为np.array格式
    y_ = np.array(y_)  # 变为np.array格式
    # arr.astype(“具体的数据类型”) 转换numpy数组的数据类型
    y_ = y_.astype(np.int64)  # 变为64位整型
    return x, y_  # 返回输入特征x，标签y_

# 判断训练集输入特征x_train和标签y_train、测试集输入特征x_test和标签y_test是否已存在
if os.path.exists(x_train_savepath) and os.path.exists(y_train_savepath) and os.path.exists(
        x_test_savepath) and os.path.exists(y_test_savepath):
    print('-------------Load Datasets-----------------')    # 若存在，直接读取
    x_train_save = np.load(x_train_savepath)
    y_train = np.load(y_train_savepath)
    x_test_save = np.load(x_test_savepath)
    y_test = np.load(y_test_savepath)
    x_train = np.reshape(x_train_save, (len(x_train_save), 28, 28))
    x_test = np.reshape(x_test_save, (len(x_test_save), 28, 28))
else:   # 若不存在，调用generateds(path, txt)函数制作数据集
    print('-------------Generate Datasets-----------------')
    x_train, y_train = generateds(train_path, train_txt)
    x_test, y_test = generateds(test_path, test_txt)
    print('-------------Save Datasets-----------------')
    x_train_save = np.reshape(x_train, (len(x_train), -1))
    x_test_save = np.reshape(x_test, (len(x_test), -1))
    # np.save(文件保存路径, 需要保存的数组)  以.npy格式将数组保存到二进制文件中
    np.save(x_train_savepath, x_train_save)
    np.save(y_train_savepath, y_train)
    np.save(x_test_savepath, x_test_save)
    np.save(y_test_savepath, y_test)

model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_freq=1)

model.summary()