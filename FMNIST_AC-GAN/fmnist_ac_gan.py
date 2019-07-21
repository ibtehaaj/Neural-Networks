import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import Conv2D, Conv2DTranspose, LeakyReLU
from keras.layers import BatchNormalization, Dropout, Embedding
from keras.layers import Activation, Concatenate
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.datasets import fashion_mnist
from keras.utils import plot_model


def define_discriminator(in_shape=(28, 28, 1), n_classes=10):
  init = RandomNormal(stddev=0.02)
  
  in_image = Input(shape=in_shape)
  
  fe = Conv2D(32, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(in_image)
  fe = LeakyReLU(alpha=0.2)(fe)
  fe = Dropout(0.5)(fe)
  
  fe = Conv2D(64, (3, 3), padding='same', kernel_initializer=init)(fe)
  fe = BatchNormalization()(fe)
  fe = LeakyReLU(alpha=0.2)(fe)
  fe = Dropout(0.5)(fe)
  
  fe = Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(fe)
  fe = BatchNormalization()(fe)
  fe = LeakyReLU(alpha=0.2)(fe)
  fe = Dropout(0.5)(fe)
  
  fe = Conv2D(256, (3,3), padding='same', kernel_initializer=init)(fe)
  fe = BatchNormalization()(fe)
  fe = LeakyReLU(alpha=0.2)(fe)
  fe = Dropout(0.5)(fe)
  
  fe = Flatten()(fe)
  
  out1 = Dense(1, activation='sigmoid')(fe)
  out2 = Dense(n_classes, activation='softmax')(fe)
  
  model = Model(in_image, [out1, out2])
  opt = Adam(lr=0.0002, beta_1=0.5)
  model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=opt)
  
  return model


def define_generator(latent_dim, n_classes=10):
  init = RandomNormal(stddev=0.02)
  
  in_label = Input(shape=(1,))
  
  li = Embedding(n_classes, 50)(in_label)
  n_nodes = 7 * 7
  li = Dense(n_nodes, kernel_initializer=init)(li)
  li = Reshape((7, 7, 1))(li)
  
  in_lat = Input(shape=(latent_dim,))
  
  n_nodes = 384 * 7 * 7
  gen = Dense(n_nodes, kernel_initializer=init)(in_lat)
  gen = Activation('relu')(gen)
  gen = Reshape((7, 7, 384))(gen)
  
  merge = Concatenate()([gen, li])
  
  gen = Conv2DTranspose(192, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init)(merge)
  gen = BatchNormalization()(gen)
  gen = Activation('relu')(gen)
  
  gen = Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init)(gen)
  out_layer = Activation('tanh')(gen)
  
  model = Model([in_lat, in_label], out_layer)
  
  return model


def define_gan(g_model, d_model):
  d_model.trainable = False
  gan_output = d_model(g_model.output)
  
  model = Model(g_model.input, gan_output)
  opt = Adam(lr=0.0002, beta_1=0.5)
  model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=opt)
  
  return model


discriminator = define_discriminator()
plot_model(discriminator, to_file='drive/My Drive/Colab Notebooks/discriminator.png', show_shapes=True, show_layer_names=False)

latent_dim = 100
generator = define_generator(latent_dim)
plot_model(generator, to_file='drive/My Drive/Colab Notebooks/generator.png', show_shapes=True, show_layer_names=False)

gan = define_gan(generator, discriminator)
plot_model(gan, to_file='drive/My Drive/Colab Notebooks/gan.png', show_shapes=True, show_layer_names=False)


def load_real_samples():
  (train_x, train_y), (_, _) = fashion_mnist.load_data()
  
  x = np.expand_dims(train_x, axis=-1)
  x = x.astype('float32')
  x = (x - 127.5) / 127.5
  
  return [x, train_y]


def generate_real_samples(dataset, n_samples):
  images, labels = dataset
  ix = np.random.randint(0, images.shape[0], n_samples)
  
  x, labels = images[ix], labels[ix]
  y = np.ones((n_samples, 1))
  
  return [x, labels], y


def generate_latent_points(latent_dim, n_samples, n_classes=10):
  x_input = np.random.randn(latent_dim * n_samples)
  
  z_input = x_input.reshape(n_samples, latent_dim)
  labels = np.random.randint(0, n_classes, n_samples)
  
  return [z_input, labels]


def generate_fake_samples(generator, latent_dim, n_samples):
  z_input, labels_input = generate_latent_points(latent_dim, n_samples)
  
  images = generator.predict([z_input, labels_input])
  y = np.zeros((n_samples, 1))
  
  return [images, labels_input], y


def summarize_performance(step, g_model, latent_dim, n_samples=100):
  [x, _], _ = generate_fake_samples(g_model, latent_dim, n_samples)
  x = (x + 1) / 2.0
  
  for i in range(100):
    plt.subplot(10, 10, 1 + i)
    plt.axis('off')
    plt.imshow(x[i, :, :, 0], cmap='gray_r')
    
  file1 = f'generated_plot_{step + 1}.png'
  plt.savefig('drive/My Drive/Colab Notebooks/' + file1)
  plt.close()
  
  file2 = f'model_{step + 1}.h5'
  g_model.save('drive/My Drive/Colab Notebooks/' + file2)
  
  print('>>> Saved: %s and %s' % (file1, file2))


def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=64):
  bat_per_epo = int(dataset[0].shape[0] / n_batch)
  n_steps = bat_per_epo * n_epochs
  half_batch = int(n_batch / 2)
  
  for i in range(n_steps):
    [x_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
    _, d_r1, d_r2 = d_model.train_on_batch(x_real, [y_real, labels_real])
    
    [x_fake, labels_fake], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
    _, d_f, d_f2 = d_model.train_on_batch(x_fake, [y_fake, labels_fake])
    
    [z_input, z_labels] = generate_latent_points(latent_dim, n_batch)
    y_gan = np.ones((n_batch, 1))
    _, g_1, g_2 = gan_model.train_on_batch([z_input, z_labels], [y_gan, z_labels])
    
    print('>%d, dr[%.3f,%.3f], df[%.3f,%.3f], g[%.3f,%.3f]' % (i+1, d_r1,d_r2, d_f,d_f2, g_1,g_2))
    
    if (i + 1) % (bat_per_epo * 10) == 0:
      summarize_performance(i, g_model, latent_dim)


latent_dim = 100
dataset = load_real_samples()

discriminator = define_discriminator()
generator = define_generator(latent_dim)
gan_model = define_gan(generator, discriminator)

train(generator, discriminator, gan_model, dataset, latent_dim)
