import os
import cv2
import string
import mediapipe as mp
import numpy as np
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")


interpreter = tf.lite.Interpreter(model_path=os.path.join("train", "models", "asl_alphabet_neuralnetwork.tflite"))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

DRAW_HANDS = True


def get_landmarks(landmarks, is_right):
    landmark_tensor = []

    for i in range(21):
        landmark_tensor += [
            landmarks[i].x if is_right else 1-landmarks[0].x,
            landmarks[i].y,
            landmarks[i].z
        ]

    return np.array(landmark_tensor).reshape(1, -1)


def predict(input_data):
    interpreter.set_tensor(input_details[0]['index'], input_data.astype("float32"))
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    return string.ascii_uppercase[np.argmax(output_data[0])]


cap = cv2.VideoCapture(0)
with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            # print("Ignoring empty camera frame.")
            continue

        # To improve performance, optionally mark the image as not writeable to pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = cv2.flip(image, 1)
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if DRAW_HANDS:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                is_right = results.multi_handedness[0].classification[0].label == 'Right'
                prediction = predict(get_landmarks(hand_landmarks.landmark, is_right))
        else:
            prediction = ""

        cv2.putText(image, prediction, (45, 375), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 10)
        cv2.putText(image, prediction, (45, 375), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)

        cv2.imshow('ASL Alphabet', image)

        if cv2.waitKey(1) == ord('q'):
            cap.release()
            cv2.destroyWindow("ASL Alphabet")
            break

cap.release()
