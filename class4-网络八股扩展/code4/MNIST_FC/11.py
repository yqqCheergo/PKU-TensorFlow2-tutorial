from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

fig = plt.figure(figsize=(20, 2))
plt.set_cmap('gray')
for i in range(0, len(x_train[:12])):
    ax = fig.add_subplot(1, 12, i + 1)
    ax.imshow(x_train[:12][i])
fig.suptitle('Subset of Original Training Images', fontsize=20)
plt.show()