import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.optimizers import Adam

plt.style.use('ggplot')

cur_time = str(time.time()).replace('.', '_')

if not os.path.exists(f"models\\{cur_time}"):
    print(f'[INFO] Creating new directory: models\\{cur_time}')
    os.makedirs(f"models\\{cur_time}")

name = f"models\\{cur_time}\\model"

data = np.load('final_data.npy', allow_pickle=True)
print('[INFO] Data Loaded! ')

X = []
y = []
for feature, label in data:
    X.append(feature)
    y.append(label)

X = np.array(X)
y = np.array(y)

ohe = OneHotEncoder()
y = ohe.fit_transform(y.reshape(-1, 1)).toarray()

print(f'[INFO] Categories:\n\t{ohe.categories_[0]}')
print(f'[INFO] Shape of X: {X.shape}')
print(f'[INFO] Shape of y: {y.shape}')

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

print(f'[INFO] Shape of x_train: {x_train.shape}')
print(f'[INFO] Shape of y_train: {y_train.shape}')
print(f'[INFO] Shape of x_test: {x_test.shape}')
print(f'[INFO] Shape of y_test: {y_test.shape}')


print('[INFO] Implementing SMOTE')
sm = SMOTE()
x_train, y_train = sm.fit_resample(x_train, y_train)

print(f'[INFO] Shape of new x_train: {x_train.shape}')
print(f'[INFO] Shape of new y_train: {y_train.shape}')

es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
tb = TensorBoard(log_dir=f'models\\{cur_time}\\logs')

model = Sequential()

model.add(Dense(256, input_shape=(257,), activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(y_train.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
              optimizer=Adam(lr=1e-3, epsilon=1e-5))

plot_model(model, to_file=f'{name}_graph.png', show_shapes=True, dpi=200)

print(f'[INFO] Model Summary:\n{model.summary()}')

print('[INFO] Training model ')
history = model.fit(x_train, y_train, batch_size=32, epochs=100,
                    callbacks=[es, tb], validation_data=(x_test, y_test))

loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f'[INFO] Test Loss: {loss}')
print(f'[INFO] Test Accuracy: {accuracy}')

model.save(f"{name}_model.h5")
print('[INFO] Model Saved!')

fig = plt.figure(figsize=(10, 10))

plt.subplot(211)
plt.plot(history.history['accuracy'], color='b', label='Acc')
plt.plot(history.history['val_accuracy'], '--', color='r', label='Val Acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(212)
plt.plot(history.history['loss'], color='b', label='Loss')
plt.plot(history.history['val_loss'], '--', color='r', label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.savefig(f'{name}_history.png')
plt.show()

predicted_classes = model.predict_classes(x_test)
matrix = confusion_matrix(np.argmax(y_test, axis=1), predicted_classes)

fig = plt.figure(figsize=(10, 10))
ax = sns.heatmap(matrix, annot=True, cmap='coolwarm')

ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(ohe.categories_[0], fontsize=8)
ax.yaxis.set_ticklabels(ohe.categories_[0], rotation=0, fontsize=8)
plt.savefig(f'{name}_cm.png')
plt.show()

cr = classification_report(np.argmax(y_test, axis=1), predicted_classes,
                           target_names=ohe.categories_[0], output_dict=True)
df = pd.DataFrame(cr).transpose()

fig = plt.figure(figsize=(10, 10))
ax = sns.heatmap(df, annot=True, cmap='coolwarm')
ax.set_title('Classification Report')
plt.savefig(f'{name}_cr.png')
plt.show()

print('[INFO] DONE')
