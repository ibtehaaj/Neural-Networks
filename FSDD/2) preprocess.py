import cv2
import numpy as np
import os
from random import shuffle

path = "Datasets\\FSDD"
path_0 = path + '\\0'
path_1 = path + '\\1'
path_2 = path + '\\2'
path_3 = path + '\\3'
path_4 = path + '\\4'
path_5 = path + '\\5'
path_6 = path + '\\6'
path_7 = path + '\\7'
path_8 = path + '\\8'
path_9 = path + '\\9'

training_data = []

for img in os.listdir(path_0):
    print(img)
    label = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    img = cv2.imread(os.path.join(path_0, img), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (20, 129))

    if img is not None:
        training_data.append([np.array(img), label])
    else:
        pass

print('Finished 0')

for img in os.listdir(path_1):
    print(img)
    label = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    img = cv2.imread(os.path.join(path_1, img), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (20, 129))

    if img is not None:
        training_data.append([np.array(img), label])
    else:
        pass

print('Finished 1')

for img in os.listdir(path_2):
    print(img)
    label = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    img = cv2.imread(os.path.join(path_2, img), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (20, 129))

    if img is not None:
        training_data.append([np.array(img), label])
    else:
        pass

print('Finished 2')

for img in os.listdir(path_3):
    print(img)
    label = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    img = cv2.imread(os.path.join(path_3, img), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (20, 129))

    if img is not None:
        training_data.append([np.array(img), label])
    else:
        pass

print('Finished 3')

for img in os.listdir(path_4):
    print(img)
    label = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    img = cv2.imread(os.path.join(path_4, img), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (20, 129))

    if img is not None:
        training_data.append([np.array(img), label])
    else:
        pass

print('Finished 4')

for img in os.listdir(path_5):
    print(img)
    label = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    img = cv2.imread(os.path.join(path_5, img), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (20, 129))

    if img is not None:
        training_data.append([np.array(img), label])
    else:
        pass

print('Finished 5')

for img in os.listdir(path_6):
    print(img)
    label = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    img = cv2.imread(os.path.join(path_6, img), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (20, 129))

    if img is not None:
        training_data.append([np.array(img), label])
    else:
        pass

print('Finished 6')

for img in os.listdir(path_7):
    print(img)
    label = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    img = cv2.imread(os.path.join(path_7, img), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (20, 129))

    if img is not None:
        training_data.append([np.array(img), label])
    else:
        pass

print('Finished 7')

for img in os.listdir(path_8):
    print(img)
    label = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    img = cv2.imread(os.path.join(path_8, img), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (20, 129))

    if img is not None:
        training_data.append([np.array(img), label])
    else:
        pass

print('Finished 8')

for img in os.listdir(path_9):
    print(img)
    label = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    img = cv2.imread(os.path.join(path_9, img), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (20, 129))

    if img is not None:
        training_data.append([np.array(img), label])
    else:
        pass

print('Finished 9')

shuffle(training_data)
print('Saving...')
np.save('digit_data.npy', training_data)
print('Saved!')
