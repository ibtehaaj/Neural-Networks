# Classification of Dogs and cats

import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, Dropout, Dense, Flatten, MaxPooling2D
import matplotlib.pyplot as plt
from time import time

print('loading data...')
data = np.load('data.npy')
print('data loaded.')

x = np.array([i[0] for i in data])
x = x.reshape(x.shape[0], 50, 50, 1)

y = [i[1] for i in data]
y = np.array(y)

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1)

#model1
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(50, 50, 1), activation='relu', padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))

model.add(Conv2D(128, (3, 3), activation='relu', padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))

model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))

model.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

tb = TensorBoard(log_dir='logs/{}'.format(time()))

history = model.fit(train_x, train_y, verbose=2, epochs=10,
                    validation_split=0.1, callbacks=[tb])

plt.plot(history.history['loss'], label='Loss')
plt.legend()
plt.show()

acc = model.evaluate(test_x, test_y, verbose=0)
print("Accuracy: %0.2f%%" % (acc[1] * 100))

