from mynet import mynet
from alexnet import alexnet
from myinception import myinception
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import random


def get_data():

    test_dir = 'F:/Gautam/Tech Stuff/Python Projects/Datasets/CIFAR10/Test'
    list_of_images = []

    for category in os.listdir(test_dir):
        sub_folder = test_dir + '/' + category
        for image in os.listdir(sub_folder):
            image_path = sub_folder + '/' + image
            list_of_images.append(image_path)

    return random.choice(list_of_images)


def model_predict(modelname, main_image):

    if modelname == mynet:
        dim = 32
        weights_path = 'F:/Gautam/Tech Stuff/Python Projects/Testing CNN/weights/mynet_cifar10-val-loss-0.85-epoch-07.h5'

    elif modelname == alexnet:
        dim = 195
        weights_path = 'F:/Gautam/Tech Stuff/Python Projects/Testing CNN/weights/alexnet_cifar10-val-loss-0.83-epoch-14.h5'

    elif modelname == myinception:
        dim = 75
        weights_path = 'F:/Gautam/Tech Stuff/Python Projects/Testing CNN/weights/myinception_cifar10-val-loss-0.69-epoch-13.h5'

    model = modelname(height=dim, width=dim, classes=10, channel=1)
    opt = Adam(lr=1e-3, decay=1e-6)

    model.compile(loss='categorical_crossentropy', optimizer=opt,
                  metrics=['accuracy'])

    model.load_weights(weights_path)

    image = cv2.resize(main_image, (dim, dim))
    image = np.array(image)
    image_NN = image.reshape(1, dim, dim, 1)
    pred = model.predict(image_NN)

    return pred


def decoding_output(pred, modelname):

    if np.argmax(pred[0]) == 0:
        print(modelname, 'Airplane')
    elif np.argmax(pred[0]) == 1:
        print(modelname, 'Automobile')
    elif np.argmax(pred[0]) == 2:
        print(modelname, 'Bird')
    elif np.argmax(pred[0]) == 3:
        print(modelname, 'Cat')
    elif np.argmax(pred[0]) == 4:
        print(modelname, 'Deer')
    elif np.argmax(pred[0]) == 5:
        print(modelname, 'Dog')
    elif np.argmax(pred[0]) == 6:
        print(modelname, 'Frog')
    elif np.argmax(pred[0]) == 7:
        print(modelname, 'Horse')
    elif np.argmax(pred[0]) == 8:
        print(modelname, 'Ship')
    elif np.argmax(pred[0]) == 9:
        print(modelname, 'Truck')


for _ in range(5):
    print('\nGetting image...')
    path = get_data()
    category = path.split('/')[7]
    main_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    print('Predicting...')

    mynet_pred = model_predict(mynet, main_image)
    alexnet_pred = model_predict(alexnet, main_image)
    myinception_pred = model_predict(myinception, main_image)

    print('\n\t\tGround Truth:', category)
    decoding_output(mynet_pred, '\t\tMynet:')
    decoding_output(alexnet_pred, '\t\tAlexnet:')
    decoding_output(myinception_pred, '\t\tMyinception:')

    # ui = input('\nPress "y" to show pic... ')

    # if ui == 'y':
    #     image = cv2.resize(main_image, (32, 32))
    #     image = np.array(image)
    #     plt.imshow(main_image, cmap=plt.cm.gray_r, interpolation='nearest')
    #     plt.title(category)
    #     plt.show()
