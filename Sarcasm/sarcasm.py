import json
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM

data = []
for line in open('Sarcasm_Headlines_Dataset.json', 'r'):
    dictionary = json.loads(line)
    data.append([dictionary['headline'], dictionary['is_sarcastic']])

data = np.array(data)

X = []
y = []

for headline, label in data:
    X.append(headline)
    y.append(label)

X = np.array(X)
y = np.array(y)
y = y.reshape(len(y), 1)

VOCAB_SIZE = 20000
SEQ_LEN = 15
DIMENSION = 50

tk = Tokenizer(num_words=VOCAB_SIZE)
tk.fit_on_texts(X)

dic = tk.word_index

with open('dictionary_sarcasm.json', 'w') as f:
    json.dump(dic, f)

sequences = tk.texts_to_sequences(X)
X = pad_sequences(sequences, maxlen=SEQ_LEN, padding='post')

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.15)

model = Sequential()
model.add(Embedding(VOCAB_SIZE, DIMENSION, input_length=SEQ_LEN))
model.add(LSTM(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])

print(model.summary())

model.fit(train_x, train_y, epochs=3, validation_split=0.1, verbose=1)

score = model.evaluate(test_x, test_y)
print('Accuracy:', score[1])
model.save('Sarcasm.model')
