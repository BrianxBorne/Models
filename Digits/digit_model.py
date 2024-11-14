import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize the images to values between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build a simple neural network model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # Flatten the 28x28 images to a 1D array
    layers.Dense(128, activation='relu'),  # Hidden layer with 128 neurons
    layers.Dense(10, activation='softmax')  # Output layer with 10 neurons (one for each digit)
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_test, y_test)
st.write(f"Test accuracy: {test_acc}")

# Function to plot the image
def plot_image(image):
    plt.imshow(image, cmap=plt.cm.binary)
    plt.axis('off')  # Turn off axis
    st.pyplot(plt)

# Create the Streamlit app
st.title("Handwritten Digit Recognition")
st.write("Upload an image of a handwritten digit (0-9) to predict.")

# File uploader to upload an image
uploaded_file = st.file_uploader("Choose an image...", type="png")

if uploaded_file is not None:
    # Read the image file
    img = plt.imread(uploaded_file)
    img = np.dot(img[...,:3], [0.299, 0.587, 0.114])  # Convert image to grayscale
    img = np.invert(np.array([img]))  # Invert the image to match MNIST style (black background, white digits)
    img = np.resize(img, (28, 28))  # Resize to match MNIST image dimensions
    img = np.expand_dims(img, axis=-1)  # Add the channel dimension
    img = img / 255.0  # Normalize the image

    # Display the uploaded image
    plot_image(img[0])

    # Predict the digit
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)

    # Show the predicted digit
    st.write(f"Predicted digit: {predicted_class}")
