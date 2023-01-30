# 数据预处理方式1：每个像素点颜色取反
from PIL import Image
import numpy as np
import tensorflow as tf

model_save_path = './checkpoint/mnist.ckpt'

model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')])
    
model.load_weights(model_save_path)    # 加载模型参数

preNum = int(input("input the number of test pictures:"))    # 询问要执行多少次图片识别任务

# 读入待识别的图片
for i in range(preNum):
    image_path = input("the path of test picture:")   # 输入待识别的图片名，如0.png
    img = Image.open(image_path)   # 打开图片
    # 训练模型时用的图片是28*28的灰度图，而输入是任意尺寸的图片，需要先resize成28*28的标准尺寸
    img = img.resize((28, 28), Image.LANCZOS)  # Image.LANCZOS是PIL内置的一种插值算法
    img_arr = np.array(img.convert('L'))   # 转换为灰度图

    # 预处理方法1：应用程序的输入图片是白底黑字，而训练模型时用的数据集是黑底白字灰度图
    img_arr = 255 - img_arr   # 每个像素点颜色取反，使得输入的从未见过的图片满足了神经网络模型对输入风格的要求
                
    img_arr = img_arr / 255.0   # 归一化
    # print("img_arr:",img_arr.shape)   # (28,28)
    '''
    由于神经网络训练时都是按batch送入网络的，所以进入predict函数前，先要把img_arr前面添加一个维度，
    从28行28列的二维数据变为一个28行28列的三维数据，送入predict预测
    '''
    x_predict = img_arr[tf.newaxis, ...]
    # print("x_predict:",x_predict.shape)   # (1,28,28)
    result = model.predict(x_predict)   # 预测
    
    pred = tf.argmax(result, axis=1)   # 输出最大的概率值
    
    print('\n')
    tf.print(pred)   # 返回预测结果