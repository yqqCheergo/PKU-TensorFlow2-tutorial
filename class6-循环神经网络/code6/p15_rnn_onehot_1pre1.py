# 用RNN实现输入一个字母，预测下一个字母
# 字母使用独热码编码
import numpy as np
import tensorflow as tf
from keras.layers import Dense, SimpleRNN
import matplotlib.pyplot as plt
import os

input_word = "abcde"
w_to_id = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}  # 单词映射到数值id的词典

id_to_onehot = {0: [1., 0., 0., 0., 0.], 1: [0., 1., 0., 0., 0.], 2: [0., 0., 1., 0., 0.], 3: [0., 0., 0., 1., 0.],
                4: [0., 0., 0., 0., 1.]}  # id编码为one-hot

# 输入特征a，对应标签b；输入特征b，对应标签c...以此类推
x_train = [id_to_onehot[w_to_id['a']], id_to_onehot[w_to_id['b']], id_to_onehot[w_to_id['c']],
           id_to_onehot[w_to_id['d']], id_to_onehot[w_to_id['e']]]
y_train = [w_to_id['b'], w_to_id['c'], w_to_id['d'], w_to_id['e'], w_to_id['a']]

# 打乱顺序
np.random.seed(7)
np.random.shuffle(x_train)
np.random.seed(7)
np.random.shuffle(y_train)
tf.random.set_seed(7)

# 使x_train符合SimpleRNN的输入要求：[送入样本数，循环核时间展开步数，每个时间步输入特征个数]
# 此处整个数据集送入，故送入样本数为len(x_train)=5；
# 输入1个字母出结果，故循环核时间展开步数为1；
# 表示为独热码有5个输入特征，故每个时间步输入特征个数为5
x_train = np.reshape(x_train, (len(x_train), 1, 5))
y_train = np.array(y_train)   # 把y_train变为numpy格式

# 构建模型
model = tf.keras.Sequential([
    SimpleRNN(3),   # 搭建具有3个记忆体的循环层（记忆体个数越多，记忆力越好，但是占用资源会更多）
    Dense(5, activation='softmax')   # 全连接，实现了输出层yt的计算；由于要映射到独热码编码，找到输出概率最大的字母，故为5
])

# 配置训练方法
model.compile(optimizer=tf.keras.optimizers.Adam(0.01),   # 学习率
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

# 断点续训
checkpoint_save_path = "./checkpoint/rnn_onehot_1pre1.ckpt"

if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 monitor='loss')  # 由于fit没有给出测试集，不计算测试集准确率，根据loss，保存最优模型

# 执行反向传播，训练参数矩阵
history = model.fit(x_train, y_train, batch_size=32, epochs=100, callbacks=[cp_callback])

# 打印网络结构，统计参数数目
model.summary()

# 提取参数
# print(model.trainable_variables)
file = open('./rnn_onehot_1pre1_weights.txt', 'w')  # 参数提取
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

###############################################    show   ###############################################

# 显示训练集和验证集的acc和loss曲线
acc = history.history['sparse_categorical_accuracy']
loss = history.history['loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.title('Training Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.title('Training Loss')
plt.legend()
plt.show()

############### predict #############

# 展示预测效果
preNum = int(input("input the number of test alphabet:"))   # 先输入要执行几次预测任务
for i in range(preNum):
    alphabet1 = input("input test alphabet:")   # 输入一个字母
    alphabet = [id_to_onehot[w_to_id[alphabet1]]]   # 把这个字母转换为独热码
    # 使alphabet符合SimpleRNN输入要求：[送入样本数，循环核时间展开步数，每个时间步输入特征个数]
    # 此处验证效果送入了1个样本，送入样本数为1；
    # 输入1个字母出结果，所以循环核时间展开步数为1；
    # 表示为独热码有5个输入特征，每个时间步输入特征个数为5
    alphabet = np.reshape(alphabet, (1, 1, 5))

    result = model.predict([alphabet])   # 得到预测结果
    pred = tf.argmax(result, axis=1)   # 选出预测结果最大的一个
    pred = int(pred)
    tf.print(alphabet1 + '->' + input_word[pred])   # input_word = "abcde"