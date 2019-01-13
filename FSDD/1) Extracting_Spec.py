from scipy import signal
from scipy.io import wavfile
import os
import scipy

path = "Datasets\\FSDD"

for sample_no, file in enumerate(os.listdir(path + '\\recordings')):

    try:
        sample_rate, samples = wavfile.read(path + '\\recordings\\' + file)
        freq, time, spec = signal.spectrogram(samples, sample_rate)
        number = file.split('_')[0]
        filename = number + '_' + str(sample_no + 1) + '.png'
        scipy.misc.imsave(path + '\\' + number + '\\' + filename, spec)

        print('Saved!', file+1)

    except Exception as e:
        pass
