# -*- coding: UTF-8 -*-
# 利用鸢尾花数据集，实现前向传播、反向传播，可视化loss曲线

# 导入所需模块
import tensorflow as tf
from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np

# 导入数据，分别为输入特征和标签
x_data = datasets.load_iris().data  # 150行4列
y_data = datasets.load_iris().target

# 随机打乱数据（因为原始数据是顺序的，顺序不打乱会影响准确率）
# seed: 随机数种子，是一个整数，当设置之后，每次生成的随机数都一样（为方便教学，以保每位同学结果一致）
np.random.seed(116)  # 使用相同的seed，保证打乱顺序后输入特征和标签仍然一一对应
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)
tf.random.set_seed(116)  # 保证每次运行这个代码文件的结果跟上次运行的结果一样

# 将打乱后的数据集分割为训练集和测试集，训练集为前120行，测试集为后30行
x_train = x_data[:-30]
y_train = y_data[:-30]
x_test = x_data[-30:]
y_test = y_data[-30:]

# 转换x的数据类型，否则后面矩阵相乘时会因数据类型不一致报错
# tf.cast(张量名,dtype=数据类型) 强制将Tensor转换为该数据类型
x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)

# 使输入特征和标签值一一对应（把数据集分批次，每个批次batch组数据）
# tf.data.Dataset.from_tensor_slices((输入特征,标签))  配成[输入特征，标签]对，每次喂入一小撮(batch)
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)  # 每32组数据打包为一个batch
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# 定义神经网络中的所有可训练参数
# 4个输入特征，故输入层为4个输入节点；因为3分类，故输出层为3个神经元（只用一层网络，输出节点数就等于分类数）
# tf.random.truncated_normal(维度,mean=均值,stddev=标准差)   默认均值为0、标准差为1。数据一定在两倍标准差内，数据更向均值集中
# 用tf.Variable()标记参数可训练
# 使用seed使每次生成的随机数相同（方便教学，使大家结果都一致，在现实使用时不写seed）
w1 = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1, seed=1))
b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=1))

lr = 0.1  # 学习率为0.1
train_loss_results = []  # 将每轮的loss记录在此列表中，为后续画loss曲线提供数据
test_acc = []  # 将每轮的acc记录在此列表中，为后续画acc曲线提供数据
epoch = 500  # 循环500轮
loss_all = 0  # 每轮分4个step，loss_all记录4个step生成的4个loss的和

# 训练部分（嵌套循环迭代，with结构更新参数，显示当前loss）
for epoch in range(epoch):  # 数据集级别的循环，每个epoch循环一次数据集
    # enumerate(列表名)  枚举出每一个元素，并在元素前配上对应的索引号，组合为：索引 元素。常在for循环中使用
    for step, (x_train, y_train) in enumerate(train_db):  # batch级别的循环，每个step循环一个batch
        '''
        在with结构中计算前向传播的预测结果y，计算损失函数loss；
        loss分别对参数w1和b1计算偏导数，更新参数w1和b1的值，打印出这一轮epoch后的损失函数值
        '''
        with tf.GradientTape() as tape:  # with结构记录梯度信息
            y = tf.matmul(x_train, w1) + b1  # 神经网络乘加运算
            y = tf.nn.softmax(y)  # 使输出y符合概率分布（此操作后与独热码同量级，可相减求loss）
            y_ = tf.one_hot(y_train, depth=3)  # 将标签值转换为独热码格式，方便计算loss和accuracy
            loss = tf.reduce_mean(tf.square(y_ - y))  # 采用均方误差损失函数mse = mean(sum(y-out)^2)
            loss_all += loss.numpy()  # 将每个step计算出的loss累加，为后续求loss平均值提供数据，这样计算的loss更准确
        # 计算loss对各个参数的梯度
        grads = tape.gradient(loss, [w1, b1])   # 嵌套循环loss对w1和b1求偏导

        # 实现梯度更新 w1 = w1 - lr * w1_grad    b = b - lr * b_grad
        # 每个step更新参数w1和b1
        w1.assign_sub(lr * grads[0])  # 参数w1自更新
        b1.assign_sub(lr * grads[1])  # 参数b自更新

    # 每个epoch，打印loss信息
    '''
    因为训练集有120组数据，batch=32，每个step只能喂入32组数据，
    需要batch级别循环4次，所以loss/4，求得每次step迭代的平均loss
    '''
    print("Epoch {}, loss: {}".format(epoch, loss_all/4))
    train_loss_results.append(loss_all / 4)  # 将4个step的loss求平均，记录在此变量中
    loss_all = 0  # loss_all归零，为记录下一个epoch的loss做准备

    # 测试部分（计算当前参数前向传播后的准确率，显示当前准确率acc）
    '''
    希望每个epoch循环后可以显示当前模型的效果，即识别准确率，
    故在epoch循环中又嵌套了一个batch级别的循环
    '''
    # total_correct为预测对的样本个数, total_number为测试的总样本数，将这两个变量都初始化为0
    total_correct, total_number = 0, 0
    # 测试时会遍历测试集中的所有数据
    for x_test, y_test in test_db:
        # 使用更新后的参数进行预测
        y = tf.matmul(x_test, w1) + b1  # 计算前向传播的预测结果y
        y = tf.nn.softmax(y)   # 变为概率分布
        pred = tf.argmax(y, axis=1)  # 返回y中最大值的索引，即预测的分类（axis=1为横向）
        pred = tf.cast(pred, dtype=y_test.dtype)   # 将pred转换为y_test即标签的数据类型
        correct = tf.cast(tf.equal(pred, y_test), dtype=tf.int32)  # 若分类正确，则correct=1，否则为0，将bool型的结果转换为int型
        correct = tf.reduce_sum(correct)  # 将每个batch的correct数加起来
        total_correct += int(correct)  # 将所有batch中的correct数加起来
        total_number += x_test.shape[0]  # total_number为测试的总样本数，也就是x_test的行数，shape[0]返回变量的行数
    # 总的准确率等于total_correct/total_number
    acc = total_correct / total_number
    test_acc.append(acc)
    print("Test_acc:", acc)
    print("--------------------------")

# 绘制 loss 曲线
plt.title('Loss Function Curve')  # 图片标题
plt.xlabel('Epoch')  # x轴变量名称
plt.ylabel('Loss')  # y轴变量名称
plt.plot(train_loss_results, label="$Loss$")  # 逐点画出trian_loss_results值并连线，连线图标是Loss
plt.legend()  # 画出曲线图标
plt.show()  # 画出图像

# 绘制 Accuracy 曲线
plt.title('Acc Curve')  # 图片标题
plt.xlabel('Epoch')  # x轴变量名称
plt.ylabel('Acc')  # y轴变量名称
plt.plot(test_acc, label="$Accuracy$")  # 逐点画出test_acc值并连线，连线图标是Accuracy
plt.legend()
plt.show()