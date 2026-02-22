import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from dataloader import get_dataloader

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

import cv2
from dataloader import ImageToTensorPreprocessor, label_to_position
from model import PalmModel

model = PalmModel()
model.load_state_dict(torch.load("fc_model.pth"))
model.to(device)
model.eval()

# todo: detect hands independently (right hand closed, left hand opened)
cap = cv2.VideoCapture(0)
find_hands = ImageToTensorPreprocessor(output_format="landmarks", static_image_mode=False)
while True:
    success, annotated_img = cap.read()
    annotated_img = cv2.flip(annotated_img, 1)
    key = cv2.waitKey(1) & 0xFF
    landmarks = find_hands(annotated_img)
    if landmarks is not None:
        landmarks = landmarks.to(device)
        classification = model(landmarks)
        predicted_label = torch.argmax(classification, dim=-1).item()
        predicted_position = label_to_position(predicted_label)
        confidence = torch.softmax(classification, dim=-1).max().item() * 100

        annotation = ""
        if predicted_position == "OPEN":
            annotation = "Open"
        else:
            annotation = "Closed"
        annotated_img = find_hands.draw_hand_landmarks(
            annotated_img,
            text = f"{annotation} ({confidence:.1f}%)"
        )
    cv2.imshow("Image", annotated_img)
    if key == ord("q"):
        break