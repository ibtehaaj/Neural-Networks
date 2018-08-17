import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from keras.models import load_model

train_dir = 'fashion-mnist_train.csv'
test_dir = 'fashion-mnist_test.csv' 


def preprocess(test=False):
    
    if test:
        df = pd.read_csv(test_dir)
    else:
        df = pd.read_csv(train_dir)
        
    df = pd.get_dummies(df, columns=['label'])

    x = df[df.columns[0:784]].values
    y = df[df.columns[784:]].values

    x = x.astype('float32')/255
    y = y.astype('float32')

    print('Shape of X: ', x.shape)
    print('Shape of y: ', y.shape)

    return x, y


test_x, test_y = preprocess(test=True)

model = load_model('fmnist_91.model')
acc = model.evaluate(test_x, test_y, verbose=2)
print('Accuracy: %0.2f%%' % (acc[1]*100))

img = test_x.reshape(test_x.shape[0], 28, 28)
print('Shape of image: ', img.shape)

while True:
    num = random.randint(0, 10000)
    pred = model.predict(test_x[num].reshape((1, 784)))

    if np.argmax(test_y[num]) == 0:
        label = 'T-shirt/Top'
    elif np.argmax(test_y[num]) == 1:
        label = 'Trouser'
    elif np.argmax(test_y[num]) == 2:
        label = 'Pullover'
    elif np.argmax(test_y[num]) == 3:
        label = 'Dress'
    elif np.argmax(test_y[num]) == 4:
        label = 'Coat'
    elif np.argmax(test_y[num]) == 5:
        label = 'Sandal'
    elif np.argmax(test_y[num]) == 6:
        label = 'Shirt'
    elif np.argmax(test_y[num]) == 7:
        label = 'Sneaker'
    elif np.argmax(test_y[num]) == 8:
        label = 'Bag'
    elif np.argmax(test_y[num]) == 9:
        label = 'Ankle boot'

    if np.argmax(pred[0]) == 0:
        prediction = 'T-shirt/Top'
    elif np.argmax(pred[0]) == 1:
        prediction = 'Trouser'
    elif np.argmax(pred[0]) == 2:
        prediction = 'Pullover'
    elif np.argmax(pred[0]) == 3:
        prediction = 'Dress'
    elif np.argmax(pred[0]) == 4:
        prediction = 'Coat'
    elif np.argmax(pred[0]) == 5:
        prediction = 'Sandal'
    elif np.argmax(pred[0]) == 6:
        prediction = 'Shirt'
    elif np.argmax(pred[0]) == 7:
        prediction = 'Sneaker'
    elif np.argmax(pred[0]) == 8:
        prediction = 'Bag'
    elif np.argmax(pred[0]) == 9:
        prediction = 'Ankle boot'

    plt.imshow(img[num], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Predicted to be an {}.\n Actual: {}'.format(prediction, label))
    plt.show()
