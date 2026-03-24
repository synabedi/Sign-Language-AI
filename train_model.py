import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

DATA_PATH = "data"
signs = ["HELLO", "YES", "NO", "COME", "OK", "SPIN", "CLAP", "UP", "DOWN", "GO_AWAY"]
no_sequences = 200
sequence_length = 20

label_map = {label:num for num, label in enumerate(signs)}
sequences, labels = [], []
print("--- 📦 LOADING DATASET ---")

for sign in signs:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            img_path = os.path.join(DATA_PATH, sign, f"clip_{sequence}", f"{frame_num}.jpg")
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) 
            if img is None: continue 
            img = cv2.resize(img, (64, 64))
            window.append(img)
        
        if len(window) == 20:
            sequences.append(window)
            labels.append(label_map[sign])

X = np.array(sequences)
X = np.expand_dims(X, axis=-1) / 255.0 
y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
model = Sequential([
    TimeDistributed(Conv2D(32, (3,3), activation='relu'), input_shape=(20, 64, 64, 1)),
    TimeDistributed(BatchNormalization()),
    TimeDistributed(MaxPooling2D(pool_size=(2,2))),
    
    TimeDistributed(Conv2D(64, (3,3), activation='relu')),
    TimeDistributed(MaxPooling2D(pool_size=(2,2))),
    TimeDistributed(Flatten()),
    LSTM(128, return_sequences=True, activation='tanh'),
    LSTM(64, return_sequences=False, activation='tanh'),
    Dense(64, activation='relu'),
    Dropout(0.5), 
    Dense(len(signs), activation='softmax')
])

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

print("--- 🧠 TRAINING STARTING ---")
model.fit(X_train, y_train, epochs=70, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stop, reduce_lr])

model.save('action_model.h5')
print("--- ✅ MODEL SAVED AS action_model.h5 ---")