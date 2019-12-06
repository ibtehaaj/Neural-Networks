import numpy as np
from tqdm import tqdm
from random import shuffle
import os

data = []

for classes in os.listdir('STFT'):
    for file in tqdm(os.listdir(os.path.join('STFT', classes))):
        temp_file = np.load(os.path.join('STFT', classes, file))
        data.append([temp_file, classes])

shuffle(data)
np.save('final_data.npy', data)
print('[INFO] Final Data Saved! ')
print(f'[INFO] Length of final data: {len(data)}')
