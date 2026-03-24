import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 1. SETUP - MUST MATCH YOUR TRAINING LIST
signs = ["HELLO", "YES", "NO", "COME", "OK", "SPIN", "CLAP", "UP", "DOWN", "GO_AWAY"]
model = load_model('action_model.h5')

sequence = []
threshold = 0.8  # AI must be 80% sure

cap = cv2.VideoCapture(0)
start_x, start_y, box_size = 150, 50, 400

print("--- 🎥 BIG TEXT INTERPRETER ACTIVE ---")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    
    # --- FIX: Define default color so the script doesn't crash ---
    box_color = (255, 0, 0) # Blue by default
    
    # Process ROI for AI
    roi = frame[start_y:start_y+box_size, start_x:start_x+box_size]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    res = cv2.resize(gray, (64, 64))
    
    sequence.append(res)
    sequence = sequence[-20:] # Sliding window
    
    if len(sequence) == 20:
        input_data = np.expand_dims(sequence, axis=0)
        input_data = np.expand_dims(input_data, axis=-1)
        input_data = input_data / 255.0
        
        res = model.predict(input_data, verbose=0)[0]
        prediction_idx = np.argmax(res)
        confidence = res[prediction_idx]
        current_sign = signs[prediction_idx]

        # --- BIG UI OVERLAY ---
        if confidence > threshold:
            # Draw a solid black bar at the bottom for readability
            cv2.rectangle(frame, (0, 420), (640, 480), (0, 0, 0), -1)
            
            # THE BIG TEXT: SIGN NAME (Green)
            cv2.putText(frame, f"{current_sign.upper()}", (20, 465), 
                        cv2.FONT_HERSHEY_DUPLEX, 2.0, (0, 255, 255), 4)
            
            # THE BIG PERCENTAGE (White)
            cv2.putText(frame, f"{int(confidence*100)}%", (480, 465), 
                        cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 2)
            
            # BOX TURNS GREEN when it knows the sign
            box_color = (0, 255, 0)

        # Visual Confidence Bar (Top of the box)
        bar_width = int(confidence * box_size)
        cv2.rectangle(frame, (start_x, start_y - 20), (start_x + bar_width, start_y - 5), box_color, -1)

    # Draw the Interaction Box (Now crash-proof!)
    cv2.rectangle(frame, (start_x, start_y), (start_x+box_size, start_y+box_size), box_color, 3)
    
    cv2.imshow('SIGN AI - BIG VIEW', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()