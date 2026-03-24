# 🤟 Real-Time ASL Sign Language Interpreter
### 🤖 Powered by CNN & macOS Native Voice

This project is a real-time American Sign Language (ASL) interpreter that uses a **Convolutional Neural Network (CNN)** to recognize gestures and provide **instant voice feedback** on macOS.

---

## 🌟 Key Features
* **98% Model Accuracy:** Optimized CNN architecture for precise gesture recognition.
* **Sequence-Based Detection:** Processes a 20-frame buffer to ensure smooth, non-flickering results.
* **Native Voice Feedback:** Integrated with the macOS `say` command via multi-threading for zero-lag audio.
* **Smart Synchronization:** Built-in `speech_lock` prevents audio overlapping and ensures the text matches the voice.

---

## 🛠️ Technical Specifications
| Component | Technology |
| :--- | :--- |
| **Deep Learning** | TensorFlow / Keras (CNN) |
| **Computer Vision** | OpenCV (Grayscale & ROI Processing) |
| **Audio Engine** | macOS Native Speech Synthesis |
| **Optimization** | Python Multi-threading & NumPy |



---

## 📊 Supported Gestures
The system currently recognizes and speaks the following **10 signs**:
1. `HELLO` 2. `YES` 3. `NO` 4. `COME` 5. `OK`
6. `SPIN` 7. `CLAP` 8. `UP` 9. `DOWN` 10. `GO_AWAY`

---

## 🚀 Installation & Usage
1. **Clone the Project:**
   ```bash
   git clone [https://github.com/your-username/signproject.git](https://github.com/your-username/signproject.git)
   cd signproject