# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import os

# Load and preprocess the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0  # Normalize pixel values

# Define the CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))  # 10 output neurons for 10 classes in CIFAR-10

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model (adjust epochs based on your needs)
history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))  # Corrected validation data

# Show the model summary to display all layers
print("\nModel Summary:")
model.summary()

# Evaluate the model's accuracy on test data
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"\nTest accuracy: {test_acc}")

# Save the model
model.save('cnn_model.keras')  # Save the model to h5 format

# Load an image for prediction
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(32, 32))
    img_array = image.img_to_array(img) / 255.0  # Normalize
    return np.expand_dims(img_array, axis=0)  # Expand dimensions for batch

# Predict and display custom images
def predict_custom_images(image_paths):
    class_names = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]

    plt.figure(figsize=(10, 4))
    for i, img_path in enumerate(image_paths):
        img_array = load_and_preprocess_image(img_path)
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])

        # Display the image and prediction
        plt.subplot(1, len(image_paths), i + 1)
        img = image.load_img(img_path, target_size=(32, 32))
        plt.imshow(img)
        plt.title(f"Prediction: {class_names[predicted_class]}")
        plt.axis('off')

    plt.show()

# Local image paths
cat_image_path = '/Users/shadow/Desktop/assignment 3 AIP/cat_image.jpg'  # Update with your cat image file
dog_image_path = '/Users/shadow/Desktop/assignment 3 AIP/dog_image.jpg'  # Update with your dog image file
car_image_path = '/Users/shadow/Desktop/assignment 3 AIP/car_image.jpg'  # Update with your car image file

# Run predictions on custom images
print("\nPredictions on custom images:")
predict_custom_images([cat_image_path, dog_image_path, car_image_path])

# Plot training & validation accuracy and loss
def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

# Display the graphs
plot_training_history(history)
