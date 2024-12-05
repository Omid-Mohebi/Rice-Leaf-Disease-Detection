import kagglehub
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


path = kagglehub.dataset_download("nirmalsankalana/rice-leaf-disease-image")
class_names = os.listdir(path)

images = []
labels = []

for class_name in class_names:
    class_folder = os.path.join(path, class_name)

    for filename in os.listdir(class_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(class_folder, filename)

            img = Image.open(img_path)
            img = img.convert('RGB')
            img = img.resize((128, 128))
            img_array = np.array(img)

            images.append(img_array)
            labels.append(class_name)

images_array = np.array(images)
labels_array = np.array(labels)

print("Images array shape:", images_array.shape)
print("Labels array shape:", labels_array.shape)


plt.imshow(images_array[0])

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels_array)

images_array, y_encoded = shuffle(images_array, y_encoded)

X_train, X_test, y_train, y_test = train_test_split(images_array, y_encoded, test_size=0.33, random_state=42)

X_train, X_test = X_train / 255, X_test / 255

from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.models import Model

K = len(set(y_train))

i = Input(shape = X_train[0].shape)
x = Conv2D(32, (3, 3), strides = 2, activation = 'relu') (i)
x = Conv2D(64, (3, 3), strides = 2, activation = 'relu') (x)
x = Conv2D(128, (3, 3), strides = 2, activation = 'relu') (x)
x = Flatten() (x)
x = Dropout(0.5) (x)
x = Dense(1024, activation = 'relu') (x)
x = Dropout(0.2) (x)
x = Dense(K, activation = 'softmax') (x)

model = Model(i, x)


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
res = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=15)

plt.plot(res.history['loss'], label='loss')
plt.plot(res.history['val_loss'], label='val_loss')
plt.legend()

plt.plot(res.history['accuracy'], label='accuracy')
plt.plot(res.history['val_accuracy'], label='val_accuracy')
plt.legend()


