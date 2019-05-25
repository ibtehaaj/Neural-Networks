import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

main_dir = 'F:/Gautam/.../Testing CNN/logs'

# mynet
filename = main_dir + '/' + 'mynet_log.csv'
df_mynet = pd.read_csv(filename)
df_mynet = df_mynet[:-5]

# myinception
filename = main_dir + '/' + 'myinception_log.csv'
df_myinception = pd.read_csv(filename)
df_myinception = df_myinception[:-5]

# alexnet
filename = main_dir + '/' + 'alexnet_log.csv'
df_alexnet = pd.read_csv(filename)
df_alexnet = df_alexnet[:-5]

fig = plt.figure(figsize=(20, 15))

ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

# accuracy
ax1.plot(df_mynet['epoch'], df_mynet['acc'], label='mynet')
ax1.plot(df_myinception['epoch'], df_myinception['acc'], label='myinception')
ax1.plot(df_alexnet['epoch'], df_alexnet['acc'], label='alexnet')
ax1.legend()
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')

# validation accuracy
ax2.plot(df_mynet['epoch'], df_mynet['val_acc'], label='mynet')
ax2.plot(df_myinception['epoch'], df_myinception['val_acc'], label='myinception')
ax2.plot(df_alexnet['epoch'], df_alexnet['val_acc'], label='alexnet')
ax2.legend()
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Validation Accuracy')

# loss
ax3.plot(df_mynet['epoch'], df_mynet['loss'], label='mynet')
ax3.plot(df_myinception['epoch'], df_myinception['loss'], label='myinception')
ax3.plot(df_alexnet['epoch'], df_alexnet['loss'], label='alexnet')
ax3.legend()
ax3.set_xlabel('Epochs')
ax3.set_ylabel('Loss')

# validation loss
ax4.plot(df_mynet['epoch'], df_mynet['val_loss'], label='mynet')
ax4.plot(df_myinception['epoch'], df_myinception['val_loss'], label='myinception')
ax4.plot(df_alexnet['epoch'], df_alexnet['val_loss'], label='alexnet')
ax4.legend()
ax4.set_xlabel('Epochs')
ax4.set_ylabel('Validation Loss')

plt.tight_layout()
plt.savefig('Training Graph.png', aspect='auto')
