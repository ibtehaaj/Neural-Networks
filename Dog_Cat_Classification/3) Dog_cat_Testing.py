# Classification of Dogs and cats
import numpy as np
import random
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.model_selection import train_test_split

print('Loading data...')
data = np.load('data.npy')
print('Data loaded.')

x = np.array([i[0] for i in data])
x = x.reshape(x.shape[0], 50, 50, 1)

y = [i[1] for i in data]
y = np.array(y)

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1,
                                                    random_state=42)

model = load_model('dog_cat.model')
print('Model Loaded Sucessfully.')

img = test_x.reshape(test_x.shape[0], 50, 50)

while True:
    num = random.randint(0, 2495)
    pred = model.predict(test_x[num].reshape(1, 50, 50, 1))

    if np.argmax(test_y[num]) == 0:
        label = 'Cat'
    elif np.argmax(test_y[num]) == 1:
        label = 'Dog'

    if np.argmax(pred[0]) == 0:
        prediction = 'Cat'
    elif np.argmax(pred[0]) == 1:
        prediction = 'Dog'

    plt.imshow(img[num], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Predicted to be an {}.\n Actual: {}'.format(prediction, label))
    plt.show()
