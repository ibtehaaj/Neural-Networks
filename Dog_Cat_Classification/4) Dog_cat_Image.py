# Classification of Dogs and cats
import numpy as np
from keras.models import load_model
import cv2

model = load_model('Dog_cat.model')
print('Model Loaded Sucessfully.')

while True:
    filename = input('\nPlease enter the file name along with ext. >>> ')

    path = 'X:/Users/...../Desktop/{}'.format(filename)

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (50, 50))
    img = np.array(img)

##    cv2.imshow('Image', img)
##    cv2.waitKey(0)
##    cv2.destroyAllWindows()

    img = img.reshape(1, 50, 50, 1)
    pred = model.predict(img)

    if np.argmax(pred[0]) == 0:
        print('The given image is of a CAT!\n')
    elif np.argmax(pred[0]) == 1:
        print('The given image is of a DOG!\n')

    print('Percentage of the image containing a CAT is', (pred[0][0] * 100))
    print('Percentage of the image containing a DOG is', (pred[0][1] * 100))

    ui = input("\nPress ENTER to continue, 'q' to exit...")

    if ui == 'q':
        print('Exiting...')
        break
