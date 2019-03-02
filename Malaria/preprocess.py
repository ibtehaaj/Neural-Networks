import cv2
import numpy as np
import os
from random import shuffle

para_dir = 'F:\\Gautam\\Tech Stuff\\Python Projects\\Datasets\\cell_images_malaria\\Parasitized'
unif_dir = 'F:\\Gautam\\Tech Stuff\\Python Projects\\Datasets\\cell_images_malaria\\Uninfected'

training_data = []

for img in os.listdir(para_dir):

    label = [1, 0]
    path = os.path.join(para_dir, img)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is not None:
        img = cv2.resize(img, (50, 50))
        training_data.append([np.array(img), label])
    else:
        pass

shuffle(training_data)
print('Parasitized Dataset done!')

for img in os.listdir(unif_dir):

    label = [0, 1]
    path = os.path.join(unif_dir, img)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is not None:
        img = cv2.resize(img, (50, 50))
        training_data.append([np.array(img), label])
    else:
        pass

shuffle(training_data)
print('Uninfected Dataset done!')

np.save('data.npy', training_data)
print('Data Saved!')
