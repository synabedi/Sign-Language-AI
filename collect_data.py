import cv2
import os
import numpy as np
DATA_PATH = "data"
signs = ["HELLO", "YES", "NO", "COME", "OK", "SPIN", "CLAP", "UP", "DOWN", "GO_AWAY"]
no_sequences = 30 
sequence_length = 20 

cap = cv2.VideoCapture(0)
start_x, start_y, box_size = 150, 50, 400

for sign in signs:
    for sequence in range(no_sequences):
        os.makedirs(os.path.join(DATA_PATH, sign, f"clip_{sequence}"), exist_ok=True)

print("--- 🖐️ NEW COLLECTION STARTING ---")

for sign in signs:
    for sequence in range(no_sequences):
        for i in range(3, 0, -1):
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            cv2.rectangle(frame, (start_x, start_y), (start_x+box_size, start_y+box_size), (0, 0, 255), 2)
            cv2.putText(frame, f"PREPARE: {sign}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Clip #{sequence}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
            cv2.putText(frame, str(i), (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 4)
            
            cv2.imshow('Hand Data Collector', frame)
            cv2.waitKey(1000) 
        for frame_num in range(sequence_length):
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            roi = frame[start_y:start_y+box_size, start_x:start_x+box_size]
            img_path = os.path.join(DATA_PATH, sign, f"clip_{sequence}", f"{frame_num}.jpg")
            cv2.imwrite(img_path, roi)
            cv2.rectangle(frame, (start_x, start_y), (start_x+box_size, start_y+box_size), (0, 255, 0), 2)
            cv2.putText(frame, f"RECORDING {sign}...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Hand Data Collector', frame)
            cv2.waitKey(50)
    print(f"✅ FINISHED COLLECTING: {sign}")
cap.release()
cv2.destroyAllWindows()
print("--- 🎉 ALL DATA COLLECTED! ---")