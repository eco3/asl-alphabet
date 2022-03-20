import os
import string

import cv2
import pandas as pd
import mediapipe as mp

import logging
from alive_progress import alive_bar
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

NO_WORKERS = 14

INPUT_FOLDER = "data_images"
OUTPUT_FOLDER = "extracted_data"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def extract_landmarks(letter):
    mp_hands = mp.solutions.hands

    path = os.path.join(INPUT_FOLDER, letter)
    image_files = [os.path.join(path, file) for file in os.listdir(path)]
    image_files_len = len(image_files)

    landmarks_export = []
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
        for index, file in enumerate(image_files):
            image = cv2.flip(cv2.imread(file), 1)
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if not results.multi_hand_landmarks:
                continue

            landmarks = results.multi_hand_landmarks[0].landmark
            is_right = results.multi_handedness[0].classification[0].label == 'Right'

            landmarks_export.append({
                "WRIST_X": landmarks[0].x if is_right else 1-landmarks[0].x,
                "WRIST_Y": landmarks[0].y,
                "WRIST_Z": landmarks[0].z,
                "THUMB_CMC_X": landmarks[1].x if is_right else 1-landmarks[1].x,
                "THUMB_CMC_Y": landmarks[1].y,
                "THUMB_CMC_Z": landmarks[1].z,
                "THUMB_MCP_X": landmarks[2].x if is_right else 1-landmarks[2].x,
                "THUMB_MCP_Y": landmarks[2].y,
                "THUMB_MCP_Z": landmarks[2].z,
                "THUMB_IP_X": landmarks[3].x if is_right else 1-landmarks[3].x,
                "THUMB_IP_Y": landmarks[3].y,
                "THUMB_IP_Z": landmarks[3].z,
                "THUMB_TIP_X": landmarks[4].x if is_right else 1-landmarks[4].x,
                "THUMB_TIP_Y": landmarks[4].y,
                "THUMB_TIP_Z": landmarks[4].z,
                "INDEX_FINGER_MCP_X": landmarks[5].x if is_right else 1-landmarks[5].x,
                "INDEX_FINGER_MCP_Y": landmarks[5].y,
                "INDEX_FINGER_MCP_Z": landmarks[5].z,
                "INDEX_FINGER_PIP_X": landmarks[6].x if is_right else 1-landmarks[6].x,
                "INDEX_FINGER_PIP_Y": landmarks[6].y,
                "INDEX_FINGER_PIP_Z": landmarks[6].z,
                "INDEX_FINGER_DIP_X": landmarks[7].x if is_right else 1-landmarks[7].x,
                "INDEX_FINGER_DIP_Y": landmarks[7].y,
                "INDEX_FINGER_DIP_Z": landmarks[7].z,
                "INDEX_FINGER_TIP_X": landmarks[8].x if is_right else 1-landmarks[8].x,
                "INDEX_FINGER_TIP_Y": landmarks[8].y,
                "INDEX_FINGER_TIP_Z": landmarks[8].z,
                "MIDDLE_FINGER_MCP_X": landmarks[9].x if is_right else 1-landmarks[9].x,
                "MIDDLE_FINGER_MCP_Y": landmarks[9].y,
                "MIDDLE_FINGER_MCP_Z": landmarks[9].z,
                "MIDDLE_FINGER_PIP_X": landmarks[10].x if is_right else 1-landmarks[10].x,
                "MIDDLE_FINGER_PIP_Y": landmarks[10].y,
                "MIDDLE_FINGER_PIP_Z": landmarks[10].z,
                "MIDDLE_FINGER_DIP_X": landmarks[11].x if is_right else 1-landmarks[11].x,
                "MIDDLE_FINGER_DIP_Y": landmarks[11].y,
                "MIDDLE_FINGER_DIP_Z": landmarks[11].z,
                "MIDDLE_FINGER_TIP_X": landmarks[12].x if is_right else 1-landmarks[12].x,
                "MIDDLE_FINGER_TIP_Y": landmarks[12].y,
                "MIDDLE_FINGER_TIP_Z": landmarks[12].z,
                "RING_FINGER_MCP_X": landmarks[13].x if is_right else 1-landmarks[13].x,
                "RING_FINGER_MCP_Y": landmarks[13].y,
                "RING_FINGER_MCP_Z": landmarks[13].z,
                "RING_FINGER_PIP_X": landmarks[14].x if is_right else 1-landmarks[14].x,
                "RING_FINGER_PIP_Y": landmarks[14].y,
                "RING_FINGER_PIP_Z": landmarks[14].z,
                "RING_FINGER_DIP_X": landmarks[15].x if is_right else 1-landmarks[15].x,
                "RING_FINGER_DIP_Y": landmarks[15].y,
                "RING_FINGER_DIP_Z": landmarks[15].z,
                "RING_FINGER_TIP_X": landmarks[16].x if is_right else 1-landmarks[16].x,
                "RING_FINGER_TIP_Y": landmarks[16].y,
                "RING_FINGER_TIP_Z": landmarks[16].z,
                "PINKY_MCP_X": landmarks[17].x if is_right else 1-landmarks[17].x,
                "PINKY_MCP_Y": landmarks[17].y,
                "PINKY_MCP_Z": landmarks[17].z,
                "PINKY_PIP_X": landmarks[18].x if is_right else 1-landmarks[18].x,
                "PINKY_PIP_Y": landmarks[18].y,
                "PINKY_PIP_Z": landmarks[18].z,
                "PINKY_DIP_X": landmarks[19].x if is_right else 1-landmarks[19].x,
                "PINKY_DIP_Y": landmarks[19].y,
                "PINKY_DIP_Z": landmarks[19].z,
                "PINKY_TIP_X": landmarks[20].x if is_right else 1-landmarks[20].x,
                "PINKY_TIP_Y": landmarks[20].y,
                "PINKY_TIP_Z": landmarks[20].z,
                "LETTER": letter,
            })

            if index % 100 == 0:
                logger.info(f" {letter}: working on file no. {index:04}/{image_files_len:04}")

    export_path = os.path.join(OUTPUT_FOLDER, f"{letter}.csv")
    pd.DataFrame(landmarks_export).to_csv(export_path)

    logger.info(f" {letter}: finished {image_files_len:04}/{image_files_len:04}")


if __name__ == '__main__':
    alphabet = string.ascii_uppercase.replace("J", "").replace("Z", "")

    with alive_bar(len(alphabet)) as bar:
        with ThreadPoolExecutor(max_workers=NO_WORKERS) as pool:
            futures = [pool.submit(extract_landmarks, l) for l in alphabet]
            for result in as_completed(futures):
                bar()
