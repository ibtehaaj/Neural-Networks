import time
from myinception import myinception
from mynet import mynet
from alexnet import alexnet
from keras.callbacks import CSVLogger
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

# mynet
dim = 32
batch_size = 32

model = mynet(height=dim, width=dim, classes=10, channel=1)
opt = Adam(lr=1e-3, decay=1e-6)

model.compile(loss='categorical_crossentropy', optimizer=opt,
              metrics=['accuracy'])

datagen = ImageDataGenerator(rescale=1. / 255)

train_dir = 'F:/Gautam/Tech Stuff/Python Projects/Datasets/CIFAR10/Train'
test_dir = 'F:/Gautam/Tech Stuff/Python Projects/Datasets/CIFAR10/Test'

train_generator = datagen.flow_from_directory(train_dir,
                                              target_size=(dim, dim),
                                              color_mode='grayscale',
                                              batch_size=batch_size,
                                              class_mode='categorical',
                                              seed=42)

validation_generator = datagen.flow_from_directory(test_dir,
                                                   target_size=(dim, dim),
                                                   color_mode='grayscale',
                                                   batch_size=batch_size,
                                                   class_mode='categorical',
                                                   seed=42)

print(train_generator.class_indices)

log = CSVLogger('logs/mynet_log.csv', append=True)

mc = ModelCheckpoint('weights/mynet_cifar10-val-loss-{val_loss:.2f}-epoch-{epoch:02d}.h5',
                     verbose=1, save_best_only=True, save_weights_only=True,
                     monitor='val_loss')

es = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

start = time.time()

model.fit_generator(train_generator, steps_per_epoch=46370 // batch_size,
                    epochs=100, validation_data=validation_generator,
                    validation_steps=3630 // batch_size,
                    callbacks=[log, mc, es])

with open('Time_taken.txt', 'a') as f:
    f.write(f'Mynet model took {time.time() - start} seconds.\n')

# alexnet
dim = 195
batch_size = 32

model = alexnet(height=dim, width=dim, classes=10, channel=1)
opt = Adam(lr=1e-3, decay=1e-6)

model.compile(loss='categorical_crossentropy', optimizer=opt,
              metrics=['accuracy'])

datagen = ImageDataGenerator(rescale=1. / 255)

train_dir = 'F:/Gautam/Tech Stuff/Python Projects/Datasets/CIFAR10/Train'
test_dir = 'F:/Gautam/Tech Stuff/Python Projects/Datasets/CIFAR10/Test'

train_generator = datagen.flow_from_directory(train_dir,
                                              target_size=(dim, dim),
                                              color_mode='grayscale',
                                              batch_size=batch_size,
                                              class_mode='categorical',
                                              seed=42)

validation_generator = datagen.flow_from_directory(test_dir,
                                                   target_size=(dim, dim),
                                                   color_mode='grayscale',
                                                   batch_size=batch_size,
                                                   class_mode='categorical',
                                                   seed=42)

print(train_generator.class_indices)

log = CSVLogger('logs/alexnet_log.csv', append=True)

mc = ModelCheckpoint('weights/alexnet_cifar10-val-loss-{val_loss:.2f}-epoch-{epoch:02d}.h5',
                     verbose=1, save_best_only=True, save_weights_only=True,
                     monitor='val_loss')

es = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

start = time.time()

model.fit_generator(train_generator, steps_per_epoch=46370 // batch_size,
                    epochs=100, validation_data=validation_generator,
                    validation_steps=3630 // batch_size,
                    callbacks=[log, mc, es])

with open('Time_taken.txt', 'a') as f:
    f.write(f'Alexnet model took {time.time() - start} seconds.\n')

# myinception
dim = 75
batch_size = 32

model = myinception(height=dim, width=dim, classes=10, channel=1)
opt = Adam(lr=1e-3, decay=1e-6)

model.compile(loss='categorical_crossentropy', optimizer=opt,
              metrics=['accuracy'])

datagen = ImageDataGenerator(rescale=1. / 255)

train_dir = 'F:/Gautam/Tech Stuff/Python Projects/Datasets/CIFAR10/Train'
test_dir = 'F:/Gautam/Tech Stuff/Python Projects/Datasets/CIFAR10/Test'

train_generator = datagen.flow_from_directory(train_dir,
                                              target_size=(dim, dim),
                                              color_mode='grayscale',
                                              batch_size=batch_size,
                                              class_mode='categorical',
                                              seed=42)

validation_generator = datagen.flow_from_directory(test_dir,
                                                   target_size=(dim, dim),
                                                   color_mode='grayscale',
                                                   batch_size=batch_size,
                                                   class_mode='categorical',
                                                   seed=42)

print(train_generator.class_indices)

log = CSVLogger('logs/myinception_log.csv', append=True)

mc = ModelCheckpoint('weights/myinception_cifar10-val-loss-{val_loss:.2f}-epoch-{epoch:02d}.h5',
                     verbose=1, save_best_only=True, save_weights_only=True,
                     monitor='val_loss')

es = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

start = time.time()

model.fit_generator(train_generator, steps_per_epoch=46370 // batch_size,
                    epochs=100, validation_data=validation_generator,
                    validation_steps=3630 // batch_size,
                    callbacks=[log, mc, es])

with open('Time_taken.txt', 'a') as f:
    f.write(f'Myinception model took {time.time() - start} seconds.\n')

print('Script Finished Successfully')
