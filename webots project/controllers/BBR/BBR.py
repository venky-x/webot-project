import os
import robot
import time
import random
import numpy as np
import cv2  # Ensure OpenCV is installed for image processing
from controller import GPS
from tensorflow.keras.models import load_model

# Set up relative path for the CNN model
current_dir = os.path.dirname(__file__)  # Get the directory of the current script (BBR/)
model_path = os.path.join(current_dir, '..', '..', 'worlds', 'cnn_model.keras')  # Adjusted relative path to the model
model = load_model(model_path)  # Load the model

# Define class names for CIFAR-10 classes
class_names = ["airplane", "dog", "bird", "cat", "deer",
               "automobile", "frog", "horse", "ship", "truck"]

# Initialize item encounter tracking
encountered_items = {"green": False, "blue": False, "red": False}

# Randomized movement based on distance to obstacles
def Behaviour0(robot):
    if robot.distance_range > 0.06:
        value = random.uniform(0.5, 0.9)  # Move forward
        robot.move(value, value)
    elif robot.distance_range > 0.05:
        value = random.uniform(0.3, 0.45)  # Move slower
        robot.move(value, value)
    elif robot.distance_range <= 0.05:
        robot.move(0, 0)  # Stop if too close to an obstacle

# Avoid obstacles by turning left
def Behaviour1(robot):
    robot.move_backward()
    robot.turn_left()

def preprocess_image(image):
    resized_image = cv2.resize(image, (32, 32))  # Resize to model input size
    normalized_image = resized_image / 255.0  # Normalize pixel values
    return np.expand_dims(normalized_image, axis=0)  # Expand dimensions for model input

# Function to display summary of encountered blocks
def display_encounter_summary():
    summary = "Blocks encountered: "
    encountered_colors = [color.capitalize() for color, found in encountered_items.items() if found]
    if encountered_colors:
        summary += ", ".join(encountered_colors)
    else:
        summary += "None"
    print(summary)

def updateSensors(robot):
    # Capture RGB values from the robot's camera
    red, green, blue = robot.get_camera_image(5)

    # Continuously display the current RGB values
    print(f"Current RGB values: Red={red}, Green={green}, Blue={blue}")

    # Check for red, blue, and green block encounters
    if red >= 130 and green <= 130 and blue <= 140 and not encountered_items["red"]:  # Red threshold
        print("I see red!!")
        encountered_items["red"] = True

    elif blue >= 150 and green <= 140 and red <= 150 and not encountered_items["blue"]:  # Blue threshold
        print("I see blue!!")
        encountered_items["blue"] = True

    elif green >= 150 and red >= 120 and blue >= 120 and not encountered_items["green"]:  # Green threshold
        print("I see green!!")
        encountered_items["green"] = True

    # Display the encounter summary after checking for blocks
    display_encounter_summary()

    # Image capturing and prediction for other purposes
    image = robot.camera.getImage()
    width = robot.camera.getWidth()
    height = robot.camera.getHeight()

    # Convert the image to a NumPy array
    image_np = np.frombuffer(image, dtype=np.uint8).reshape((height, width, 4))
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)  # Convert to RGB

    # Preprocess the image for prediction
    input_image = preprocess_image(image_np)

    # Make prediction using the CNN model
    predictions = model.predict(input_image)
    predicted_class = np.argmax(predictions[0])
    print(f"Prediction: {class_names[predicted_class]}")

def main():
    robot1 = robot.BR()
    robot1.init_devices()

    while True:
        try:
            robot1.reset_actuator_values()
            robot1.blink_leds()
            gps = GPS('gps')
            gps.enable(32)

            robot1.distance_range = robot1.get_sensor_input()  # Get distance range
            updateSensors(robot1)

            if robot1.front_obstacles_detected():
                Behaviour1(robot1)  # Turn left to avoid obstacles
            else:
                Behaviour0(robot1)  # Wander

            robot1.set_actuators()
            robot1.step()

        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
