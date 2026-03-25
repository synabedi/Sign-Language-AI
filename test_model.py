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
        # Clean text for natural speech (e.g., GO_AWAY -> Go Away)
        clean_text = str(text).strip().replace("_", " ")
        os.system(f'say "{clean_text}"')
        speech_lock = False
        
    threading.Thread(target=run, daemon=True).start()

# --- 2. THE FINAL SYNCED LABELS (MATCHED TO YOUR BRAIN MAP) ---
# Index 0=Hello, 1=Yes, 2=No, 3=Come, 4=Ok
# Remaining filled based on your project's 10-sign list
signs = [
    "HELLO",    # Index 0
    "YES",      # Index 1
    "NO",       # Index 2
    "COME",     # Index 3
    "OK",       # Index 4
    "SPIN",     # Index 5
    "CLAP",     # Index 6
    "UP",       # Index 7
    "DOWN",     # Index 8
    "GO_AWAY"   # Index 9
]

# --- 3. MODEL INITIALIZATION ---
model = load_model('action_model.h5')
sequence = []
threshold = 0.70  # Higher threshold for better accuracy
last_spoken = ""
last_speech_time = 0

cap = cv2.VideoCapture(0)
start_x, start_y, box_size = 150, 50, 400

print("--- 🎥 SIGN AI: PRODUCTION READY ---")
speak("System Online")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    box_color = (255, 0, 0) 
    
    # --- 4. PRE-PROCESSING ---
    roi = frame[start_y:start_y+box_size, start_x:start_x+box_size]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    res = cv2.resize(gray, (64, 64))
    
    sequence.append(res)
    sequence = sequence[-20:] # Buffer of 20 frames
    
    if len(sequence) == 20:
        input_data = np.expand_dims(sequence, axis=0)
        input_data = np.expand_dims(input_data, axis=-1)
        input_data = input_data / 255.0
        
        res = model.predict(input_data, verbose=0)[0]
        prediction_idx = np.argmax(res)
        confidence = res[prediction_idx]
        
        current_sign = signs[prediction_idx]

        # --- 5. VOICE & UI TRIGGER ---
        if confidence > threshold:
            current_time = time.time()
            
            if current_sign != last_spoken and not speech_lock:
                if (current_time - last_speech_time > 2.5): # 2.5s delay to prevent spam
                    print(f"✅ ACTION: {current_sign} ({int(confidence*100)}%)")
                    speak(current_sign)
                    last_spoken = current_sign
                    last_speech_time = current_time

            # Update Screen UI
            cv2.rectangle(frame, (0, 420), (640, 480), (0, 0, 0), -1)
            cv2.putText(frame, f"SIGN: {current_sign}", (20, 465), 
                        cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 0), 2)
            box_color = (0, 255, 0)
        else:
            last_spoken = ""
            box_color = (255, 0, 0)

    # --- 6. DISPLAY ---
    cv2.rectangle(frame, (start_x, start_y), (start_x+box_size, start_y+box_size), box_color, 3)
    cv2.imshow('ASL AI INTERPRETER', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()