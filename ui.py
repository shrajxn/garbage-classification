import gradio as gr
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import os

# Load the garbage classification model
model_path = '/content/garbage_model.h5'
model = load_model(model_path)

target_size = (224, 224)

class_labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Function to preprocess the input image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to classify garbage and provide information
def classify_garbage(img_array):
    result = model.predict(img_array)
    predicted_class = np.argmax(result)
    confidence = result[0][predicted_class] * 100
    class_name = class_labels[predicted_class]

    return class_name, confidence

# Gradio interface function
def classify_image(image_obj):
    # Save the uploaded image to a temporary directory
    uploads_dir = '/content/uploads/'
    os.makedirs(uploads_dir, exist_ok=True)

    # Extract the name of the uploaded file
    image_name = image_obj.name if hasattr(image_obj, 'name') else 'unknown_file'

    # Save and resize the image
    image_path = os.path.join(uploads_dir, f"{image_name}.jpg")
    image_obj.save(image_path)
    img_array = preprocess_image(image_path)

    # Classify the garbage and get information
    class_name, confidence = classify_garbage(img_array)

    # Display the result
    return class_name, confidence

# Gradio Interface without Submit Button
iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil", label="Upload a garbage image"),
    outputs=[gr.Label(), gr.Label()],
    live=True  # Set to False in production to disable debug mode
)

# Launch the Gradio interface
iface.launch()