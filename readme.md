# 🤟 Sign Language AI Recognition Model (CNN)
**Image-Based Gesture Detection with 98% Accuracy**

This project uses a custom **Convolutional Neural Network (CNN)** to recognize sign language gestures directly from image data. Unlike landmark-based systems, this model processes raw pixel data to identify patterns and shapes associated with different signs.

### 📊 Project Highlights
* **Architecture:** Pure Convolutional Neural Network (CNN).
* **Accuracy:** Reached a validated **98% accuracy** on the test set.
* **Tech Stack:** Python, TensorFlow, Keras, OpenCV.
* **Input:** Processed image frames (RGB/Grayscale) from a webcam.

### 📁 File Structure
* `train_model.py`: Defines the CNN architecture (Conv2D, MaxPooling, Dense layers) and trains the model.
* `test_model.py`: Captures webcam frames, pre-processes the images, and runs the CNN prediction.
* `action_model.h5`: The trained weights and architecture of your CNN.
* `.gitignore`: Configured to ignore the large image dataset and local environments.

### 🚀 How to Run
1. Clone the repository.
2. Install dependencies: `pip install tensorflow opencv-python`.
3. Run `python test_model.py`.