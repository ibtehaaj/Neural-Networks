from math import sqrt
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt


def generate_latent_points(latent_dim, n_samples, n_class):
    x_input = np.random.randn(latent_dim * n_samples)
    z_input = x_input.reshape(n_samples, latent_dim)
    labels = np.asarray([n_class for _ in range(n_samples)])

    return [z_input, labels]


def show_images(examples, n_examples):
    for i in range(n_examples):
        plt.subplot(sqrt(n_examples), sqrt(n_examples), i + 1)
        plt.axis('off')
        plt.imshow(examples[i, :, :, 0], cmap='gray_r')

    plt.show()


def get_user_data():

    print('''Choose the type of image to be generated:
          0 - T-Shirt / Top
          1 - Trouser
          2 - Pullover
          3 - Dress
          4 - Coat
          5 - Sandal
          6 - Shirt
          7 - Sneaker
          8 - Bag
          9 - Ankle Boot
          ''')

    n_class = int(input('Enter the corresponding number: '))
    n_examples = int(input('Enter a number of images (square) to be generated: '))

    return n_class, n_examples


model = load_model('model_93700.h5')

while True:
    n_class, n_examples = get_user_data()
    latent_dim = 100
    latent_points, labels = generate_latent_points(latent_dim, n_examples, n_class)
    x = model.predict([latent_points, labels])
    x = (x + 1) / 2.0

    show_images(x, n_examples)
