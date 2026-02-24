import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from custom_dataset import get_dataloader

import sys
sys.path.insert(0, '..')
import frontend
from custom_dataset import get_dataloader
import pandas as pd
import cv2
from model import PalmModel

# CUDA: Nvidia CUDA-enabled GPU
if torch.cuda.is_available():
    print("Using GPU")
    device = torch.device("cuda")

# MPS: Apple Silicon
elif torch.backends.mps.is_available():
    print("Using MPS")
    device = torch.device("mps")

# CPU: 
else:
    print("Using CPU")
    device = torch.device("cpu")


from pathlib import Path
src_directory = Path("custom_dataset")
label_map_path = src_directory / "label_map.csv"


label_map = pd.read_csv(label_map_path)
label_to_name = dict(zip(label_map["label"].astype(int), label_map["gesture_name"]))
name_to_label = dict(zip(label_map["gesture_name"], label_map["label"].astype(int)))

#label_df = pd.read_csv("labels.csv")
#label_to_gesture = dict(zip(label_df['label'], label_df['gesture_name']))
def main():

    model = PalmModel()
    model.load_state_dict(torch.load("fc_model.pth"))
    model.to(device)
    model.eval()

    # todo: detect hands independently (right hand closed, left hand opened)
    webcam = cv2.VideoCapture(0)
    landmarker = frontend.MediaPipeLandmarker()
    gesture_detection = frontend.GestureDetection()
    while True:
        success, webcam_frame = webcam.read()
        #webcam_frame = cv2.flip(webcam_frame, 1)
        key = cv2.waitKey(1) & 0xFF

        height, width, n_channels = webcam_frame.shape
        handedness, hand_landmarks = landmarker(webcam_frame)
        overlay_mask = np.zeros((height, width, n_channels), dtype="uint8")


        if handedness is not None:
            lh_tensor = gesture_detection.mediapipe_to_tensor(handedness, hand_landmarks, isolated_hand="Left")
            lh_tensor = gesture_detection.expand_one_hand_to_two_hands(lh_tensor, isolated_hand="Left")

            # classify left hand
            lh_tensor = lh_tensor.to(device)
            classification = model(lh_tensor)
            predicted_label = torch.argmax(classification, dim=-1).item()
            predicted_gesture = label_to_name.get(predicted_label, f"Unknown ({predicted_label})")
            confidence = torch.softmax(classification, dim=-1).max().item() * 100

            landmarker.draw_overlay_hands(
                webcam_frame,
                overlay_mask,
                text_lr = (f"{predicted_gesture} ({confidence:.1f}%)", "")
            )


        webcam_frame = cv2.flip(webcam_frame, 1)
        webcam_frame = cv2.add(webcam_frame, overlay_mask)
        cv2.imshow("Image", webcam_frame)
        if key == ord("q"):
            break

if __name__ == "__main__":
    main()