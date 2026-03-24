🤟 Sign Language AI Interpreter (macOS)
A real-time Computer Vision application that uses a Convolutional Neural Network (CNN) to translate American Sign Language (ASL) gestures into text and synchronized speech on macOS.

🌟 Project Highlights
High Accuracy: Optimized CNN model achieving 98% validation accuracy.

Real-Time Translation: Processes live video at 30+ FPS with a 20-frame sequence buffer for smooth detection.

Native Voice Integration: Uses a multi-threaded os.system bridge to trigger the native macOS "Siri" voice without lagging the video feed.

State-Lock Logic: Custom synchronization logic to prevent "audio stuttering" and overlapping speech.

🛠️ Technical Stack
Deep Learning: TensorFlow / Keras (CNN)

Computer Vision: OpenCV (ROI processing, Grayscale conversion)

Speech: macOS Native Speech Synthesis (say command)

Development: Python 3.9+, NumPy, Threading

📊 Supported Gestures
The model is trained to recognize and speak the following 10 signs:
HELLO, YES, NO, COME, OK, SPIN, CLAP, UP, DOWN, GO_AWAY