import os
import gc
import librosa
import numpy as np
import librosa.display
from tqdm import tqdm
import matplotlib.pyplot as plt


def create_spectrogram(filename, name):
    plt.interactive(False)
    clip, sample_rate = librosa.load(filename, sr=None)
    fig = plt.figure(figsize=[0.72, 0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    plt.gray()
    plt.savefig(name, dpi=400, bbox_inches='tight', pad_inches=0)
    plt.close()
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename, name, clip, sample_rate, fig, ax, S


counter = 0
for classes in os.listdir('Birdcalls'):
    for file in tqdm(os.listdir(os.path.join('Birdcalls', classes))):
        filename = os.path.join('Birdcalls', classes, file)
        save_dir = os.path.join('Spectrograms', classes)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        name = save_dir + '\\' + file.split('.')[0] + '.jpg'
        create_spectrogram(filename, name)

        counter += 1
        if counter % 20 == 0:
            gc.collect()

print('\n\tDone!!!')
