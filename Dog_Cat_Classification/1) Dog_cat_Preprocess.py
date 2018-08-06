# Classification of Dogs and cats

import cv2
import numpy as np
import os
from random import shuffle

cat_dir = 'X:/...../Datasets/PetImages/Cat'
dog_dir = 'X:/...../Datasets/PetImages/Dog'

training_data = []

for img in os.listdir(cat_dir):
    
    label = [1, 0]
    path = os.path.join(cat_dir,img)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is not None:
        img = cv2.resize(img, (50, 50))
        training_data.append([np.array(img), label])
    else:
        pass
    
shuffle(training_data)
print('Cat dataset is done.')

for img in os.listdir(dog_dir):
    
    label = [0, 1]
    path = os.path.join(dog_dir,img)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is not None:
        img = cv2.resize(img, (50, 50))
        training_data.append([np.array(img), label])
        shuffle(training_data)
    else:
        pass

shuffle(training_data)
print('Dog dataset is done.')

np.save('data.npy', training_data)
print('Data saved.')

print('\n', training_data)
print("\nScript of preprocessing done.")
ui = input('Press enter to exit..')
