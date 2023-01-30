# 导入所需模块
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

# 读入数据/标签，生成x_train y_train
df = pd.read_csv('dot.csv')
x_data = np.array(df[['x1', 'x2']])  # 取出x1和x2两列，并转成二维数组
y_data = np.array(df['y_c'])  # 取出y_c列，并转成列表

# np.vstack(数组1,数组2)：将两个数组按垂直方向叠加
x_train = np.vstack(x_data).reshape(-1, 2)
y_train = np.vstack(y_data).reshape(-1, 1)

Y_c = [['red' if y else 'blue'] for y in y_train]   # 1的话红点，0的话蓝点
# Y_c打印出来是一个二维数组，里面是['red']['blue']这样的元素

# 转换x的数据类型，否则后面矩阵相乘时会因数据类型问题报错
x_train = tf.cast(x_train, tf.float32)
y_train = tf.cast(y_train, tf.float32)

# from_tensor_slices函数切分传入的张量的第一个维度，生成相应的数据集，使输入特征和标签值一一对应
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)

# 生成两层神经网络的参数，输入层为2个神经元，隐藏层为11个神经元（随便选的），1层隐藏层，输出层为1个神经元
# 用tf.Variable()保证参数可训练
w1 = tf.Variable(tf.random.normal([2, 11]), dtype=tf.float32)   # 2个输入特征
b1 = tf.Variable(tf.constant(0.01, shape=[11]))

w2 = tf.Variable(tf.random.normal([11, 1]), dtype=tf.float32)  # 第二层网络输入特征的个数11与第一层网络的输出个数11要一致
b2 = tf.Variable(tf.constant(0.01, shape=[1]))   # 整个神经网络的输出节点个数要和标签一样，数据集中每组x1、x2对应一个标签

# 定义超参数
lr = 0.005  # 学习率
epoch = 800  # 循环轮数

# 训练部分
# for循环嵌套，反向传播求梯度，更新参数
for epoch in range(epoch):   # epoch是对数据集的循环计数
    for step, (x_train, y_train) in enumerate(train_db):   # step是对batch的循环计数
        # 在with结构中完成前向传播计算出y
        with tf.GradientTape() as tape:  # 记录梯度信息

            h1 = tf.matmul(x_train, w1) + b1  # 记录神经网络乘加运算
            h1 = tf.nn.relu(h1)
            y = tf.matmul(h1, w2) + b2

            # 采用均方误差损失函数mse = mean(sum(y-out)^2)
            loss = tf.reduce_mean(tf.square(y_train - y))

        # 计算loss对各个参数的梯度
        variables = [w1, b1, w2, b2]
        grads = tape.gradient(loss, variables)  # 损失函数loss分别对w1/b1/w2/b2求偏导数

        # 实现梯度更新，分别更新w1/b1/w2/b2
        # w1 = w1 - lr * w1_grad
        # tape.gradient是自动求导结果与[w1, b1, w2, b2]  索引为0，1，2，3
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])

    # 每20个epoch，打印loss信息
    if epoch % 20 == 0:
        print('epoch:', epoch, 'loss:', float(loss))

# 预测部分
print("*******predict*******")

# xx在-3到3之间以步长为0.1，yy在-3到3之间以步长0.1,生成间隔数值点
xx, yy = np.mgrid[-3:3:.1, -3:3:.1]   # 生成网格坐标点，密度是0.1
'''
xx：
[[-3.  -3.  -3.  ... -3.  -3.  -3. ]
 [-2.9 -2.9 -2.9 ... -2.9 -2.9 -2.9]
 [-2.8 -2.8 -2.8 ... -2.8 -2.8 -2.8]
 ...
 [ 2.7  2.7  2.7 ...  2.7  2.7  2.7]
 [ 2.8  2.8  2.8 ...  2.8  2.8  2.8]
 [ 2.9  2.9  2.9 ...  2.9  2.9  2.9]]
'''
# 将xx , yy拉直，并合并配对为二维张量，生成二维坐标点
grid = np.c_[xx.ravel(), yy.ravel()]
'''
xx.ravel()：
[-3.  -3.  -3.  ...  2.9  2.9  2.9]

grid:
[[-3.  -3. ]
 [-3.  -2.9]
 [-3.  -2.8]
 ...
 [ 2.9  2.7]
 [ 2.9  2.8]
 [ 2.9  2.9]]
'''

grid = tf.cast(grid, tf.float32)
'''
grid：
tf.Tensor(
[[-3.  -3. ]
 [-3.  -2.9]
 [-3.  -2.8]
 ...
 [ 2.9  2.7]
 [ 2.9  2.8]
 [ 2.9  2.9]], shape=(3600, 2), dtype=float32)
'''

# 将网格坐标点喂入训练好的神经网络，神经网络会为每个坐标输出一个预测值，probs为输出
# 要区分输出偏向1还是0，可以把预测结果为0.5的线标出颜色，即为0和1的区分线
probs = []
for x_test in grid:
    # 使用训练好的参数进行预测
    h1 = tf.matmul([x_test], w1) + b1
    h1 = tf.nn.relu(h1)
    y = tf.matmul(h1, w2) + b2   # y为预测结果
    probs.append(y)

# 取第0列给x1，取第1列给x2
x1 = x_data[:, 0]
x2 = x_data[:, 1]

# probs的shape调整成xx的样子
probs = np.array(probs).reshape(xx.shape)

# 画出x1和x2的散点
plt.scatter(x1, x2, color=np.squeeze(Y_c))  # squeeze去掉纬度是1的纬度,相当于去掉[['red'],['blue']]内层括号,变为['red','blue']

# 把坐标xx yy和对应的值probs放入contour函数，给probs值为0.5的所有点上色  plt.show()后 显示的是红蓝点的分界线
'''
plt.contour(X, Y, Z, [levels])  作用：绘制轮廓线，类似于等高线
X和Y就是坐标位置  Z代表每个坐标对应的高度值，是一个二维数组
levels传入一个包含高度值的一维数组，这样便会画出传入的高度值对应的等高线
'''
plt.contour(xx, yy, probs, levels=[.5])   # 画出预测值y=0.5的曲线
plt.show()

# 读入红蓝点，画出分割线，不包含正则化   结果：轮廓不够平滑，存在过拟合现象
# 不清楚的数据，建议print出来查看