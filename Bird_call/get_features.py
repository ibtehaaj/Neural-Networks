import os
import librosa
import numpy as np
import librosa.display
from tqdm import tqdm


def get_stft(audioname, featurename):
    clip, _ = librosa.load(filename, sr=None)
    stft = np.abs(librosa.stft(clip, n_fft=512, hop_length=256, win_length=512))
    features = np.mean(stft, axis=1)
    np.save(featurename, features)


for classes in os.listdir('Birdcalls'):
    for file in tqdm(os.listdir(os.path.join('Birdcalls', classes))):
        filename = os.path.join('Birdcalls', classes, file)
        save_dir = os.path.join('STFT', classes)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        name = save_dir + '\\' + file.split('.')[0] + '.npy'
        get_stft(filename, name)

print('\n\tDone!!!')
