from keras.layers import Input, Dropout, Dense, Flatten, concatenate
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model


def myinception(height=299, width=299, channel=3, classes=1000):  # min_shape=(75, 75, 1), classes=2

    # stem
    inputs = Input(shape=(height, width, channel))

    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=2, padding='valid', activation='relu')(inputs)
    conv1 = BatchNormalization()(conv1)

    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='valid', activation='relu')(conv1)
    conv2 = BatchNormalization()(conv2)

    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(conv2)
    conv3 = BatchNormalization()(conv3)

    # left
    pool1 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(conv3)

    # right
    conv4 = Conv2D(filters=96, kernel_size=(3, 3), strides=2, padding='valid', activation='relu')(conv3)
    conv4 = BatchNormalization()(conv4)

    concat1 = concatenate([pool1, conv4], axis=3)

    # left
    conv5 = Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(concat1)
    conv5 = BatchNormalization()(conv5)

    conv6 = Conv2D(filters=96, kernel_size=(3, 3), strides=1, padding='valid', activation='relu')(conv5)
    conv6 = BatchNormalization()(conv6)

    # right
    conv7 = Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(concat1)
    conv7 = BatchNormalization()(conv7)

    conv8 = Conv2D(filters=64, kernel_size=(7, 1), strides=1, padding='same', activation='relu')(conv7)
    conv8 = BatchNormalization()(conv8)

    conv9 = Conv2D(filters=64, kernel_size=(1, 7), strides=1, padding='same', activation='relu')(conv8)
    conv9 = BatchNormalization()(conv9)

    conv10 = Conv2D(filters=96, kernel_size=(3, 3), strides=1, padding='valid', activation='relu')(conv9)
    conv10 = BatchNormalization()(conv10)

    concat2 = concatenate([conv6, conv10], axis=3)

    # left
    conv11 = Conv2D(filters=192, kernel_size=(3, 3), strides=2, padding='valid', activation='relu')(concat2)
    conv11 = BatchNormalization()(conv11)

    # right
    pool1 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(concat2)

    concat3 = concatenate([conv11, pool1], axis=3)

    # inception-A (x4)
    # 1
    avg_pool = AveragePooling2D(pool_size=(3, 3), strides=1, padding='same')(concat3)

    conv1 = Conv2D(filters=96, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(avg_pool)
    conv1 = BatchNormalization()(conv1)

    # 2
    conv2 = Conv2D(filters=96, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(concat3)
    conv2 = BatchNormalization()(conv2)

    # 3
    conv3 = Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(concat3)
    conv3 = BatchNormalization()(conv3)

    conv4 = Conv2D(filters=96, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(conv3)
    conv4 = BatchNormalization()(conv4)

    # 4
    conv5 = Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(concat3)
    conv5 = BatchNormalization()(conv5)

    conv6 = Conv2D(filters=96, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(conv5)
    conv6 = BatchNormalization()(conv6)

    conv7 = Conv2D(filters=96, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(conv6)
    conv7 = BatchNormalization()(conv7)

    concat4 = concatenate([conv1, conv2, conv4, conv7], axis=3)

    # reduction-A
    # 1
    pool1 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(concat4)

    # 2
    conv1 = Conv2D(filters=384, kernel_size=(3, 3), strides=2, padding='valid', activation='relu')(concat4)
    conv1 = BatchNormalization()(conv1)

    # 3
    conv2 = Conv2D(filters=192, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(concat4)
    conv2 = BatchNormalization()(conv2)

    conv3 = Conv2D(filters=224, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(conv2)
    conv3 = BatchNormalization()(conv3)

    conv4 = Conv2D(filters=256, kernel_size=(3, 3), strides=2, padding='valid', activation='relu')(conv3)
    conv4 = BatchNormalization()(conv4)

    concat5 = concatenate([pool1, conv1, conv4], axis=3)

    # inception-B (x7)
    # 1
    avg_pool = AveragePooling2D(pool_size=(3, 3), strides=1, padding='same')(concat5)

    conv1 = Conv2D(filters=128, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(avg_pool)
    conv1 = BatchNormalization()(conv1)

    # 2
    conv2 = Conv2D(filters=384, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(concat5)
    conv2 = BatchNormalization()(conv2)

    # 3
    conv3 = Conv2D(filters=192, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(concat5)
    conv3 = BatchNormalization()(conv3)

    conv4 = Conv2D(filters=224, kernel_size=(1, 7), strides=1, padding='same', activation='relu')(conv3)
    conv4 = BatchNormalization()(conv4)

    conv5 = Conv2D(filters=256, kernel_size=(1, 7), strides=1, padding='same', activation='relu')(conv4)
    conv5 = BatchNormalization()(conv5)

    # 4
    conv6 = Conv2D(filters=192, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(concat5)
    conv6 = BatchNormalization()(conv6)

    conv7 = Conv2D(filters=192, kernel_size=(1, 7), strides=1, padding='same', activation='relu')(conv6)
    conv7 = BatchNormalization()(conv7)

    conv8 = Conv2D(filters=224, kernel_size=(7, 1), strides=1, padding='same', activation='relu')(conv7)
    conv8 = BatchNormalization()(conv8)

    conv9 = Conv2D(filters=224, kernel_size=(1, 7), strides=1, padding='same', activation='relu')(conv8)
    conv9 = BatchNormalization()(conv9)

    conv10 = Conv2D(filters=256, kernel_size=(7, 1), strides=1, padding='same', activation='relu')(conv9)
    conv10 = BatchNormalization()(conv10)

    concat6 = concatenate([conv1, conv2, conv5, conv10], axis=3)

    # reduction-B
    # 1
    pool1 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(concat6)

    # 2
    conv1 = Conv2D(filters=192, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(concat6)
    conv1 = BatchNormalization()(conv1)

    conv2 = Conv2D(filters=192, kernel_size=(3, 3), strides=2, padding='valid', activation='relu')(conv1)
    conv2 = BatchNormalization()(conv2)

    # 3
    conv3 = Conv2D(filters=256, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(concat6)
    conv3 = BatchNormalization()(conv3)

    conv4 = Conv2D(filters=256, kernel_size=(1, 7), strides=1, padding='same', activation='relu')(conv3)
    conv4 = BatchNormalization()(conv4)

    conv5 = Conv2D(filters=320, kernel_size=(7, 1), strides=1, padding='same', activation='relu')(conv4)
    conv5 = BatchNormalization()(conv5)

    conv6 = Conv2D(filters=320, kernel_size=(3, 3), strides=2, padding='valid', activation='relu')(conv5)
    conv6 = BatchNormalization()(conv6)

    concat7 = concatenate([pool1, conv2, conv6], axis=3)

    # inception-C (x3)
    # 1
    avg_pool = AveragePooling2D(pool_size=(3, 3), strides=1, padding='same')(concat7)

    conv1 = Conv2D(filters=256, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(avg_pool)
    conv1 = BatchNormalization()(conv1)

    # 2
    conv2 = Conv2D(filters=256, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(concat7)
    conv2 = BatchNormalization()(conv2)

    # 3
    conv3 = Conv2D(filters=384, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(concat7)
    conv3 = BatchNormalization()(conv3)

    conv4 = Conv2D(filters=256, kernel_size=(1, 3), strides=1, padding='same', activation='relu')(conv3)
    conv4 = BatchNormalization()(conv4)

    # 3_1
    conv5 = Conv2D(filters=256, kernel_size=(3, 1), strides=1, padding='same', activation='relu')(conv3)
    conv5 = BatchNormalization()(conv5)

    # 4
    conv6 = Conv2D(filters=384, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(concat7)
    conv6 = BatchNormalization()(conv6)

    conv7 = Conv2D(filters=448, kernel_size=(1, 3), strides=1, padding='same', activation='relu')(conv6)
    conv7 = BatchNormalization()(conv7)

    conv8 = Conv2D(filters=512, kernel_size=(3, 1), strides=1, padding='same', activation='relu')(conv7)
    conv8 = BatchNormalization()(conv8)

    conv9 = Conv2D(filters=256, kernel_size=(3, 1), strides=1, padding='same', activation='relu')(conv8)
    conv9 = BatchNormalization()(conv9)

    # 4_1
    conv10 = Conv2D(filters=256, kernel_size=(1, 3), strides=1, padding='same', activation='relu')(conv8)
    conv10 = BatchNormalization()(conv10)

    concat8 = concatenate([conv1, conv2, conv4, conv5, conv9, conv10], axis=3)

    avg_pool = AveragePooling2D(pool_size=(8, 8), strides=1, padding='same')(concat8)

    dropped = Dropout(0.8)(avg_pool)  # paper says keep 0.8
    flattened = Flatten()(dropped)
    output = Dense(classes, activation='softmax')(flattened)

    model = Model(inputs=inputs, outputs=output, name='my_inception')

    return model
