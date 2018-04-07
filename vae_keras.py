#!/usr/bin/env python
# coding=utf-8

# refer to https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist

batch_size = 100
original_dim = 784
latent_dim = 2
intermediate_dim = 256
epochs = 50
epsilon_std = 1.0

# 输入x是图像
x = Input(shape=(original_dim,))
# encoder隐层
h = Dense(intermediate_dim, activation='relu')(x)
# 输出隐变量z的均值
z_mean = Dense(latent_dim)(h)
# 输出隐变量z的log方差
z_log_var = Dense(latent_dim)(h)

# reparameterization trick
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# decoder的隐层
decoder_h = Dense(intermediate_dim, activation='relu')
# decoder的输出层，范围为[0, 1]
decoder_mean = Dense(original_dim, activation='sigmoid')

# 构建解码层
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

# VAE模型
vae = Model(x, x_decoded_mean)

# VAE损失函数，包括decoder输出误差xent_loss与encoder输出KL误差kl_loss
# 此处xent_loss也可以用MSE，因为此处的x值是连续的
xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(xent_loss + kl_loss)

vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop')
vae.summary()


# 模型训练
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
mnist_data = np.load('data/mnist.npz')
x_train, y_train, x_test, y_test = mnist_data['x_train'], mnist_data['y_train'], \
                                    mnist_data['x_test'], mnist_data['y_test']

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

vae.fit(x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, None))

# 将输入映射为隐变量（reuse上述模型训练后的参数）
encoder = Model(x, z_mean)

# 显示隐层分布（二维分布），不同颜色代表不同类别的数字
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
plt.colorbar()
plt.show()

# 构建生成器模型（reuse上述模型训练后的参数）
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)

# display a 2D manifold of the digits
n = 15  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
# 由于z是高斯分布，可以取z在高斯分布累积概率密度函数纵坐标值在[0.05,0.95]范围的横坐标值
# z是二维的，需要在两个维度上都进行上述操作
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = generator.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()