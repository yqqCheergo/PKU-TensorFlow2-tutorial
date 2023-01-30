import tensorflow as tf
import os   # 为了判断保存的模型参数是否存在

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_save_path = "./checkpoint/mnist.ckpt"   # 定义存放模型的路径和文件名，命名为ckpt文件，生成ckpt文件时会同步生成索引表
if os.path.exists(checkpoint_save_path + '.index'):   # 通过判断是否存在索引表，判断是否已经保存过模型参数
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)   # 若已经有了索引表，直接读取模型参数

# 保存训练出来的模型参数，使用回调函数，返回给cp_callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)

history = model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_freq=1,
                    callbacks=[cp_callback])    # 在fit中加入callbacks(回调选项)，赋值给history

model.summary()

# 训练过程中出现checkpoint文件夹，里面存放的就是模型参数
# 再次运行，程序会加载刚才保存的模型参数，这次训练的准确率是在刚刚保存的模型基础上继续提升的