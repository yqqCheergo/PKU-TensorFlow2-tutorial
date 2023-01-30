import tensorflow as tf

w = tf.Variable(tf.constant(5, dtype=tf.float32))   # 设定参数w的随机初始值为5，设定为可训练
lr = 0.2
epoch = 40

for epoch in range(epoch):  # for epoch定义顶层循环，表示对数据集循环epoch次，此例数据集数据仅有1个w,初始化时constant赋值为5，循环40次迭代

    # 用with结构让损失函数loss对参数w求梯度
    with tf.GradientTape() as tape:  # with结构到grads框起了梯度的计算过程
        loss = tf.square(w + 1)   # 损失函数为(w+1)^2
    grads = tape.gradient(loss, w)  # .gradient函数告知谁对谁求导

    w.assign_sub(lr * grads)  # .assign_sub对变量做自减 即：w -= lr*grads 即 w = w - lr*grads
    print("After %s epoch,w is %f,loss is %f" % (epoch, w.numpy(), loss))   # 打印出每次训练后的参数w和损失函数loss

# lr初始值：0.2   请自改学习率  0.001  0.999 看收敛过程
# 最终目的：找到 loss 最小 即 w = -1 的最优参数w