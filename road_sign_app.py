import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import pickle
import pyttsx3
import time


model = load_model('vgg16_best_model.keras')


with open('class_indices.pkl', 'rb') as f:
    class_indices = pickle.load(f)
class_labels = {v: k for k, v in class_indices.items()}

# GTSRB HUMAN-READABLE LABELS
gtsrb_label_names = {
    '0': "Speed limit (20km/h)", '1': "Speed limit (30km/h)", '2': "Speed limit (50km/h)",
    '3': "Speed limit (60km/h)", '4': "Speed limit (70km/h)", '5': "Speed limit (80km/h)",
    '6': "End of speed limit (80km/h)", '7': "Speed limit (100km/h)", '8': "Speed limit (120km/h)",
    '9': "No passing", '10': "No passing for heavy vehicles", '11': "Right-of-way at next intersection",
    '12': "Priority road", '13': "Yield", '14': "Stop", '15': "No vehicles", '16': "Heavy vehicles prohibited",
    '17': "No entry", '18': "General caution", '19': "Left curve", '20': "Right curve",
    '21': "Double curve", '22': "Bumpy road", '23': "Slippery road", '24': "Road narrows",
    '25': "Road work", '26': "Traffic signals", '27': "Pedestrian crossing", '28': "Children crossing",
    '29': "Bicycles crossing", '30': "Beware of ice", '31': "Wild animals crossing", '32': "End of all limits",
    '33': "Turn right ahead", '34': "Turn left ahead", '35': "Ahead only", '36': "Go straight or right",
    '37': "Go straight or left", '38': "Keep right", '39': "Keep left", '40': "Roundabout",
    '41': "End of no passing", '42': "End of no passing for heavy vehicles"
}

# INITIALIZE TTS AND STATE VARIABLES
engine = pyttsx3.init()
engine.setProperty('rate', 150)

last_label = ""
last_spoken_time = 0
cooldown = 5  # seconds

prediction_buffer = []
buffer_size = 5
frame_count = 0
frame_skip = 3  # Process every 3rd frame

# START CAMERA
cap = cv2.VideoCapture(0)
print(" Press 'q' to quit")

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    label = "No sign detected"

    if frame_count % frame_skip == 0:
        img = cv2.resize(frame, (160, 160))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        pred = model.predict(img_array)[0]
        confidence = np.max(pred)
        class_id = np.argmax(pred)

        if confidence > 0.75:
            folder_id = str(class_labels[class_id])
            label = gtsrb_label_names.get(folder_id, "Unknown")
        else:
            print(f" Low confidence ({confidence:.2f}) - Ignored")

        # Add to buffer
        prediction_buffer.append(label)
        if len(prediction_buffer) > buffer_size:
            prediction_buffer.pop(0)

        # Speak if stable
        if prediction_buffer.count(label) >= 3 and label != last_label and label != "No sign detected":
            current_time = time.time()
            if current_time - last_spoken_time > cooldown:
                print(f" Speaking: {label}")
                engine.say(f"{label} detected")
                engine.runAndWait()
                last_label = label
                last_spoken_time = current_time

    # Overlay
    fps = 1.0 / (time.time() - start_time)
    cv2.putText(frame, f"Prediction: {label}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 0, 0), 2)
    cv2.imshow("Real-Time Road Sign Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# CLEANUP
engine.stop()
cap.release()
cv2.destroyAllWindows()
