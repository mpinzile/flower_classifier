import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Load the saved model
model_filename = 'flower_classifier_model.h5'
saved_model = tf.keras.models.load_model(model_filename)

# Path to the test image you want to use for testing
test_image_path = './flowers/sunflower/6953297_8576bf4ea3.jpg'

# Load and preprocess the test image
IMAGE_SIZE = 224
img = tf.keras.preprocessing.image.load_img(
    test_image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Normalize pixel values

# Predict the class of the image
predictions = saved_model.predict(img_array)
predicted_class_index = np.argmax(predictions[0])
class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
predicted_class = class_names[predicted_class_index]

# Display the image and the predicted class
plt.imshow(img)
plt.title(f"Predicted Class: {predicted_class}")
plt.show()
