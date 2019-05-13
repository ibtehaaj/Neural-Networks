import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.initializers import Constant

SEQ_LEN = 30
VOCAB_SIZE = 10000
DIMENSION = 100  # 50, 100, 200, 300

GLOVE_DIR = f'F:/.../Datasets/glove.6B/glove.6B.{DIMENSION}d.txt'
DATA_DIR = 'F:/.../Datasets/sentiment.csv'

print('Indexing word vectors.')
embedding_index = {}

with open(GLOVE_DIR, encoding='utf8') as f:
    for line in f:
        values = line.split(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs

print(f'Found {len(embedding_index)} word vectors!')

df = pd.read_csv(DATA_DIR)
X = df.text
y = df.label

tk = Tokenizer(num_words=VOCAB_SIZE)
tk.fit_on_texts(X)
word_index = tk.word_index

print(f'Training Corpus has {len(word_index)} words!')

with open('dictionary_sentiment.json', 'w') as f:
    json.dump(word_index, f)

sequences = tk.texts_to_sequences(X)
X = pad_sequences(sequences, maxlen=SEQ_LEN, padding='post')
y = y.values

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.1)

print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)

num_words = min(VOCAB_SIZE, len(word_index)) + 1
embedding_matrix = np.zeros((num_words, DIMENSION))

for word, i in word_index.items():
    if i > VOCAB_SIZE:
        continue
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

model = Sequential()

model.add(Embedding(num_words, DIMENSION, input_length=SEQ_LEN,
                    embeddings_initializer=Constant(embedding_matrix),
                    trainable=False))

model.add(LSTM(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

opt = Adam(lr=1e-3, decay=1e-5)
model.compile(loss='binary_crossentropy', optimizer=opt,
              metrics=['accuracy'])

print(model.summary())

es = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')

mc = ModelCheckpoint('SA-epoch-{epoch:02d}-loss-{val_loss:.2f}-acc-{val_acc:.2f}.h5',
                     verbose=1, period=1, save_best_only=True,
                     save_weights_only=False, monitor='val_loss')

history = model.fit(train_x, train_y, epochs=50, validation_split=0.1, verbose=1,
                    callbacks=[es, mc])

score = model.evaluate(test_x, test_y)
print('Accuracy:', score[1])

plot_model(model, to_file=f'SA-loss-{score[0]}-acc-{score[1]}.png', show_shapes=True)

pred = model.predict(test_x)

prediction = []
for i in pred:
    if i[0] <= 0.5:
        prediction.append(0)
    elif i[0] > 0.5:
        prediction.append(1)

plt.plot(history.history['acc'], color='b', label='Accuracy')
plt.plot(history.history['val_acc'], color='r', label='Val_Acc')
plt.legend()
plt.show()

plt.plot(history.history['loss'], color='b', label='Loss')
plt.plot(history.history['val_loss'], color='r', label='Val_Loss')
plt.legend()
plt.show()

confm = confusion_matrix(test_y, prediction)
sns.heatmap(confm, cmap='binary', annot=True)
plt.show()
