import os
import librosa
import numpy as np
from pydub import AudioSegment
from tensorflow.keras.models import load_model

model = load_model('models\\1575619662_1015356_92\\model_model.h5')
print('[INFO] Model Loaded...')

while True:
    try:
        filename = input('\nEnter filename of audio along with extension >>> ')
        original_path = os.path.join('Test Calls', filename)
        wav_name = filename.split('.')[0] + '_wav' + '.wav'
        wav_path = os.path.join('Test Calls', wav_name)
        sound = AudioSegment.from_file(original_path)
        sound = sound.set_channels(1)
        sound.export(wav_path, format='wav')

        clip, _ = librosa.load(wav_path, sr=None)
        stft = np.abs(librosa.stft(clip, n_fft=512, hop_length=256, win_length=512))
        features = np.mean(stft, axis=1)
        feature = np.array([features])

        print('[INFO] Data Preprocessed...')

        pred = model.predict(feature)
        pred_classes = model.predict_classes(feature)[0]

        print(f'[INFO] Probability: {pred}')

        if pred_classes == 0:
            value = 'Blue Rock Pigeon'

        elif pred_classes == 1:
            value = 'Common Tailorbird'

        elif pred_classes == 2:
            value = 'Purple Sunbird'

        print(f'[INFO] Predicted to be "{value}"')

        ui = input('\nPress ENTER to continue, "q" to quit: ')
        if ui == 'q':
            print('Exiting...')
            break

    except Exception as e:
        print(f'\n[ERROR]: {str(e)}')
