# 数据预处理方式2：让输入图片变为只有黑色和白色的高对比度图片（二值化）
from PIL import Image
import numpy as np
import tensorflow as tf

model_save_path = './checkpoint/mnist.ckpt'

model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')])

model.load_weights(model_save_path)

preNum = int(input("input the number of test pictures:"))

for i in range(preNum):
    image_path = input("the path of test picture:")
    print('\n')
    img = Image.open(image_path)
    img = img.resize((28, 28), Image.LANCZOS)
    img_arr = np.array(img.convert('L'))

    # 预处理方法2：让输入图片变为只有黑色和白色的高对比度图片，在保留图片有用信息的同时，滤去了背景噪声，图片更干净
    for i in range(28):
        for j in range(28):   # 使用嵌套for循环遍历输入图片的每一个像素点
            if img_arr[i][j] < 200:   # 把灰度值＜200的像素点变为255（纯白）     当阈值选择合理时，识别效果会更好
                img_arr[i][j] = 255
            else:   # 其余像素点变为0（纯黑）
                img_arr[i][j] = 0

    img_arr = img_arr / 255.0
    x_predict = img_arr[tf.newaxis, ...]
    result = model.predict(x_predict)

    pred = tf.argmax(result, axis=1)

    '''
    输出结果pred是张量，需要用tf.print
    print打印出来是tf.Tensor([1],shape=(1,),dtype=int64)
    '''
    tf.print(pred)