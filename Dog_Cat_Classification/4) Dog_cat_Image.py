# Classification of Dogs and cats
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.model_selection import train_test_split
import cv2

model = load_model('dog_cat.model')
print('Model Loaded Sucessfully.')

while True:
    filename = input('\nPlease enter the file name along with ext. >>> ')

    path = 'C:/Users/.../Desktop/{}'.format(filename)

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (50, 50))
    img = np.array(img)

    img_NN = img.reshape(1, 50, 50, 1)
    pred = model.predict(img_NN)

    if np.argmax(pred[0]) == 0:
        print('\n\tThe given image is of a CAT!')
    elif np.argmax(pred[0]) == 1:
        print('\n\tThe given image is of a DOG!')

    print('Percentage of the image containing a CAT is', (pred[0][0] * 100))
    print('Percentage of the image containing a DOG is', (pred[0][1] * 100))

    plt.imshow(img, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()

    ui = input("\nPress ENTER to continue, 'q' to exit...")

    if ui == 'q':
        print('Exiting...')
        break
