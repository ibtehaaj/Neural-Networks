import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, Dropout, Dense, Flatten, MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.callbacks import ReduceLROnPlateau
import time
import matplotlib.pyplot as plt

print('Loading data...')
data = np.load('digit_data.npy')
print('Data Loaded!')

x = np.array([i[0] for i in data])
x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
y = np.array([i[1] for i in data])

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2,
                                                    random_state=42)

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(129, 20, 1), activation='relu',
          padding="same"))
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

model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

mc = ModelCheckpoint('digit-{epoch:02d}-{val_loss:.2f}.model',
                     verbose=1, period=1, save_best_only=True,
                     save_weights_only=False,
                     monitor='val_loss', mode='auto')

es = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')

tb = TensorBoard(log_dir='logs/{}'.format(time.time()))

lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5,
                       min_lr=0.00001, verbose=1)

start = time.time()

history = model.fit(train_x, train_y, verbose=1, epochs=1000,
                    validation_split=0.1, callbacks=[mc, es, tb])

end = time.time()
print("Model took %0.2fs to train." % (end-start))

plt.plot(history.history['acc'], color='b', label='Accuracy')
plt.plot(history.history['val_acc'], color='r', label='Val_Acc')
plt.legend()
plt.show()

plt.plot(history.history['loss'], color='b', label='Loss')
plt.plot(history.history['val_loss'], color='r', label='Val_Loss')
plt.legend()
plt.show()
