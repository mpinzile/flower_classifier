# Image Classificaation
# David Mpinzile

# Loading Appropriate Libraries
import tensorflow as tf
import numpy as np

# Directory and image settings
BASE_DIR = './flowers'
IMAGE_SIZE = 224
BATCH_SIZE = 64

# Data augmentation for training images
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2,
                                                                horizontal_flip=True, validation_split=0.1
                                                                )

# Data augmentation for testing images
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255, validation_split=0.1)

# Load training data
train_generator = train_datagen.flow_from_directory(BASE_DIR, target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                    batch_size=BATCH_SIZE, subset='training'
                                                    )

# Load validation data
validation_generator = test_datagen.flow_from_directory(BASE_DIR, target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                        batch_size=BATCH_SIZE, subset='validation'
                                                        )

# Create a sequential model
cnn = tf.keras.Sequential()

# Add convolutional layers
cnn.add(tf.keras.layers.Conv2D(filters=64, padding='same', strides=2,
        kernel_size=3, activation='relu', input_shape=(224, 224, 3)))

# Add max pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Flatten the output
cnn.add(tf.keras.layers.Flatten())

# Add a dense layer for classification
cnn.add(tf.keras.layers.Dense(5, activation='softmax'))

# Compile the model
cnn.compile(optimizer=tf.keras.optimizers.Adam(),
            loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
cnn.fit(train_generator, epochs=10, validation_data=validation_generator)

# Save the model
model_filename = 'flower_classifier_model.h5'
cnn.save(model_filename)
print("Model saved as", model_filename)
