from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import time

plt.style.use('ggplot')

data_gen = ImageDataGenerator(rescale=1. / 255)

train_gen = data_gen.flow_from_directory('Spectrograms/Train',
                                         target_size=(64, 64),
                                         batch_size=16,
                                         class_mode='binary',
                                         color_mode='grayscale')

test_gen = data_gen.flow_from_directory('Spectrograms/Test',
                                        target_size=(64, 64),
                                        batch_size=7,
                                        class_mode='binary',
                                        color_mode='grayscale')

model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', input_shape=(64, 64, 1), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

opt = Adam(lr=0.0005)

model.compile(optimizer=opt, loss="binary_crossentropy",
              metrics=["accuracy"])

# print(model.summary())
# plot_model(model, to_file='Images\\birdcall_model.png', show_shapes=True, dpi=300)

mc = ModelCheckpoint('weights/birdcall-val_loss-{val_loss:.3f}-epoch-{epoch:02d}.model',
                     verbose=1, period=1, save_best_only=True,
                     save_weights_only=True,
                     monitor='val_loss', mode='auto')

es = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')

STEP_SIZE_TRAIN = train_gen.n // train_gen.batch_size
STEP_SIZE_TEST = test_gen.n // test_gen.batch_size

history = model.fit_generator(generator=train_gen,
                              steps_per_epoch=STEP_SIZE_TRAIN,
                              validation_data=test_gen,
                              validation_steps=STEP_SIZE_TEST,
                              callbacks=[mc, es],
                              epochs=20, verbose=1)

plt.plot(history.history['accuracy'], color='b', label='Accuracy')
plt.plot(history.history['val_accuracy'], '--', color='b', label='Val_Acc')
plt.plot(history.history['loss'], color='r', label='Loss')
plt.plot(history.history['val_loss'], '--', color='r', label='Val_Loss')
plt.title('Bird Call')
plt.xlabel('Epochs')
plt.ylabel('Acc / Loss')
plt.legend()
name = 'Images\\' + str(time.time()) + '_birdcall.png'
plt.savefig(name)
plt.show()

model.load_weights('weights\\birdcall.model')
score = model.evaluate_generator(test_gen)
print(f'Accuracy: {score[1]} Loss: {score[0]}')
