import random
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
from keras.models import load_model


def display_image(noisy, decoded, original):

    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    ax1.imshow(noisy.reshape(28, 28), cmap='gray')
    ax1.title.set_text('Noisy Image')

    ax2.imshow(decoded.reshape(28, 28), cmap='gray')
    ax2.title.set_text('Decoded by Autoencoder')

    ax3.imshow(original.reshape(28, 28), cmap='gray')
    ax3.title.set_text('Original Image')

    plt.show()


model = load_model('fmnist_deep.h5')

(_, _), (test_x, _) = fashion_mnist.load_data()
test_x = test_x.astype('float32') / 255
test_x = test_x.reshape(-1, 28, 28, 1)
noise_factor = 0.3
test_x_noisy = test_x + noise_factor * np.random.normal(loc=0.0,
                                                        scale=1,
                                                        size=test_x.shape)

test_x_noisy = np.clip(test_x_noisy, 0., 1.)

while True:
    num = random.randint(0, 10_000)
    noisy = test_x_noisy[num]
    original = test_x[num]
    decoded = model.predict(noisy.reshape(-1, 28, 28, 1))
    display_image(noisy, decoded, original)
