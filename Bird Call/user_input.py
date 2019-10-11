import os
import cv2
import librosa
import numpy as np
import librosa.display
from pydub import AudioSegment
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

try:
    filename = input('\nEnter filename of audio along with extension >>> ')
    original_path = os.path.join('User_call', filename)
    wav_name = filename.split('.')[0] + '_wav' + '.wav'
    wav_path = os.path.join('User_call', wav_name)
    fig_name = filename.split('.')[0] + '_png' + '.png'
    fig_path = os.path.join('User_call', fig_name)

    sound = AudioSegment.from_file(original_path)
    sound = sound.set_channels(1)
    sound.export(wav_path, format='wav')

    plt.interactive(False)
    clip, sample_rate = librosa.load(wav_path, sr=None)
    fig = plt.figure(figsize=[0.72, 0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    plt.gray()
    plt.savefig(fig_path, dpi=400, bbox_inches='tight', pad_inches=0)
    plt.close()
    fig.clf()
    plt.close(fig)
    plt.close('all')

except Exception as e:
    print(str(e))
    pass

print('\nData Preprocessed...')

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

model.load_weights('weights\\birdcall.model')
print('Model Loaded...\n')

img = cv2.imread(fig_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (64, 64))
img = np.array(img)
img = img / 255.
img = img.reshape(1, 64, 64, 1)
pred = model.predict(img)

print(pred)
print('\nDone')
