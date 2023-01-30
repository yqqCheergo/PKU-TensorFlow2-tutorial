# 自制数据集，构建一层神经网络，预测酸奶日销量
import tensorflow as tf
import numpy as np

SEED = 23455   # 随机种子，保证每次生成的数据集一样

rdm = np.random.RandomState(seed=SEED)  # 生成[0,1)之间的随机数
x = rdm.rand(32, 2)  # 生成32行2列的输入特征x，包含了32组0-1之间的随机数x1和x2

# .rand()生成[0,1)之间的随机数
y_ = [[x1 + x2 + (rdm.rand() / 10.0 - 0.05)] for (x1, x2) in x]  # 生成噪声[0,1)/10=[0,0.1); [0,0.1)-0.05=[-0.05,0.05)
x = tf.cast(x, dtype=tf.float32)   # x转变数据类型

w1 = tf.Variable(tf.random.normal([2, 1], stddev=1, seed=1))  # 随机初始化参数w1，初始化为两行一列

epoch = 15000   # 数据集迭代次数
lr = 0.002

for epoch in range(epoch):
    with tf.GradientTape() as tape:
        y = tf.matmul(x, w1)   # for循环中用with结构，求前向传播计算结果y
        loss_mse = tf.reduce_mean(tf.square(y_ - y))   # 求均方误差损失函数loss_mse
    grads = tape.gradient(loss_mse, w1)   # 损失函数对待训练参数w1求偏导
    w1.assign_sub(lr * grads)   # 更新参数w1

    if epoch % 500 == 0:   # 每迭代500轮数据打印当前参数w1
        print("After %d training steps,w1 is " % (epoch))
        print(w1.numpy(), "\n")
print("Final w1 is: ", w1.numpy())