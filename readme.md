# 🤟 Sign Language AI Recognition Model
**Real-time Gesture Detection with High Accuracy**

This project uses Deep Learning to recognize sign language gestures through a webcam. By leveraging Computer Vision and LSTM (Long Short-Term Memory) networks, the model can predict actions with a high degree of precision.

### 📊 Project Highlights
* **Accuracy:** Reached a validated **98% accuracy** during training.
* **Tech Stack:** Python, TensorFlow, Keras, OpenCV, and Mediapipe.
* **Data:** Trained on a custom dataset of landmark sequences (Keypoints).

### 📁 File Structure
* `train_model.py`: The script used to build and train the neural network.
* `test_model.py`: The real-time script that uses your webcam to predict gestures.
* `action_model.h5`: The saved, pre-trained "brain" of the model.
* `.gitignore`: Keeps the repository clean by excluding large datasets and environments.

### 🚀 How to Run
1. Clone the repository.
2. Install requirements (TensorFlow, OpenCV, Mediapipe).
3. Run `python test_model.py` to start the real-time detection.