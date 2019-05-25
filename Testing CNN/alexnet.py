from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization


def alexnet(height=224, width=224, channel=3, classes=1000):  # min shape=(195, 195, 1)

    model = Sequential()

    model.add(Conv2D(filters=96, input_shape=(height, width, channel),
                     kernel_size=(11, 11), strides=4, padding='valid',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid'))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=1, padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid'))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=1, padding='valid', activation='relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=1, padding='valid', activation='relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid'))
    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())

    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())

    model.add(Dense(classes, activation='softmax'))

    return model
