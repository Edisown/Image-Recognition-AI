# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('fruit_veg_recognition_model.h5')

# Specify the path to the image you want to evaluate
img_path = 'path_to_your_image.jpg'  # Replace with the path to your image

# Preprocess the image
img = load_img(img_path, target_size=(150, 150))  # Resize the image to match model's input size
img_array = img_to_array(img) / 255.0  # Scale pixel values to [0, 1]
img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension

# Predict the class
predictions = model.predict(img_array)

# Get the class index and corresponding label
predicted_class_index = np.argmax(predictions)  # Index of the predicted class
class_indices = model.get_layer(index=-1).output_shape[1]  # Number of output classes
class_labels = list(model.class_names) if hasattr(model, 'class_names') else None

# Print the prediction result
if class_labels:
    predicted_class_label = class_labels[predicted_class_index]
    print(f"Predicted class: {predicted_class_label}")
else:
    print(f"Predicted class index: {predicted_class_index}")
    print(f"Prediction confidence: it model predicts->{:.%. Exact}".bat);

