import cv2
import time
import argparse
import urllib.request
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from keras.applications.mobilenet_v2 import decode_predictions

parser = argparse.ArgumentParser('Object Detection')
parser.add_argument('--path', type=str, help='Path to image file along with extension')
parser.add_argument('--url', type=str, help='URL to image')
args = parser.parse_args()

model = MobileNetV2(weights='imagenet', input_shape=(224, 224, 3))

while True:

    if args.path:
        image_path = args.path

    elif args.url:
        filename = round(time.time())
        urllib.request.urlretrieve(f'{args.url}', f'd_{filename}.png')
        image_path = f'd_{filename}.png'

    else:
        image_path = input('\nFull Path to image file: ')

    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, 224, 224, 3))
    image = preprocess_input(image)

    prediction = model.predict(image)
    label = decode_predictions(prediction)

    for i in range(5):
        new_label = label[0][i]
        print(f'\n\t{new_label[1]} ({new_label[2] * 100})')

    if args.path or args.url:
        break

    else:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)
        plt.show()
