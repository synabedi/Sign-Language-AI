import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import threading
import time

# --- 1. MAC VOICE ENGINE ---
speech_lock = False

def speak(text):
    global speech_lock
    if speech_lock:
        return
    
    def run():
        global speech_lock
        speech_lock = True
        # Clean text: "GO_AWAY" becomes "Go Away"
        clean_text = str(text).strip().replace("_", " ")
        os.system(f'say "{clean_text}"')
        speech_lock = False
        
    threading.Thread(target=run, daemon=True).start()

# --- 2. YOUR EXACT ORDER ---
signs = ["HELLO", "YES", "NO", "COME", "OK", "SPIN", "CLAP", "UP", "DOWN", "GO_AWAY"]
model = load_model('action_model.h5')

sequence = []
threshold = 0.7  # Raising back to 0.7 for better accuracy
last_spoken = ""
last_speech_time = 0

cap = cv2.VideoCapture(0)
start_x, start_y, box_size = 150, 50, 400

print("--- 🎥 AI ONLINE: LABELS RE-ALIGNED ---")
speak("System Online")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    box_color = (255, 0, 0) # Default Blue
    
    # ROI Pre-processing
    roi = frame[start_y:start_y+box_size, start_x:start_x+box_size]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    res = cv2.resize(gray, (64, 64))
    
    sequence.append(res)
    sequence = sequence[-20:] 
    
    if len(sequence) == 20:
        input_data = np.expand_dims(sequence, axis=0)
        input_data = np.expand_dims(input_data, axis=-1)
        input_data = input_data / 255.0
        
        res = model.predict(input_data, verbose=0)[0]
        prediction_idx = np.argmax(res)
        confidence = res[prediction_idx]
        current_sign = signs[prediction_idx]

        # LIVE STATUS IN TERMINAL
        print(f"AI Thinking: {current_sign} ({int(confidence*100)}%)", end='\r')

        if confidence > threshold:
            current_time = time.time()
            
            # TRIGGER VOICE
            if current_sign != last_spoken and not speech_lock:
                if (current_time - last_speech_time > 2.0):
                    print(f"\n✅ SPEAKING: {current_sign}")
                    speak(current_sign)
                    last_spoken = current_sign
                    last_speech_time = current_time

            # UI TEXT (Matches the voice)
            cv2.rectangle(frame, (0, 420), (640, 480), (0, 0, 0), -1)
            cv2.putText(frame, f"{current_sign}", (20, 465), 
                        cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 0), 3)
            box_color = (0, 255, 0)
        else:
            # Reset if hand leaves or confidence is low
            last_spoken = ""
            box_color = (255, 0, 0)

        # Confidence Bar UI
        bar_width = int(confidence * box_size)
        cv2.rectangle(frame, (start_x, start_y - 20), (start_x + bar_width, start_y - 5), box_color, -1)

    cv2.rectangle(frame, (start_x, start_y), (start_x+box_size, start_y+box_size), box_color, 3)
    cv2.imshow('SIGN AI - FINAL CORRECTED', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()