
from google.colab import drive
drive.mount('/content/drive')

import zipfile
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Unzip the file
zip_file_path = '/content/drive/MyDrive/archive.zip'
extracted_path = '/content/drive/MyDrive/garbage/Garbage classification/'
# Check if the file exists
# if os.path.exists(zip_file_path):
#     # Create the target directory if it doesn't exist
#     os.makedirs(extracted_path, exist_ok=True)

#     # Unzip the file
#     with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
#         zip_ref.extractall(extracted_path)

#     print(f"Successfully unzipped '{zip_file_path}' to '{extracted_path}'.")
# else:
#     print(f"Error: The file '{zip_file_path}' does not exist.")

garbage_classification_dir = os.path.join('/content/drive/MyDrive/garbage/Garbage classification/Garbage classification/Garbage classification', 'Garbage classification')

img_width, img_height = 224, 224
batch_size = 32
epochs = 50
num_classes = 6

# Data augmentation for the training set
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   validation_split=0.2)

# Data augmentation for the test set (only rescaling)
test_datagen = ImageDataGenerator(rescale=1./255)

# Generate batches of augmented data for training
train_generator = train_datagen.flow_from_directory(garbage_classification_dir,
                                                    target_size=(img_width, img_height),
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    subset='training')

# Generate batches of augmented data for validation (without data augmentation)
validation_generator = train_datagen.flow_from_directory(garbage_classification_dir,
                                                         target_size=(img_width, img_height),
                                                         batch_size=batch_size,
                                                         class_mode='categorical',
                                                         subset='validation')

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model with a lower learning rate
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator,
          steps_per_epoch=train_generator.samples // batch_size,
          epochs=epochs,
          validation_data=validation_generator,
          validation_steps=validation_generator.samples // batch_size)

model.save('/content/garbage_model.h5')


