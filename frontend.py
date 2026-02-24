#import socket
#import json

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
from urllib.request import urlretrieve
import numpy as np

import torch

import pandas as pd


class MediaPipeLandmarker:
    def __init__(self):

        task_file = Path("hand_landmarker.task")

        # download hand_landmarker.task for MediaPipe
        if not task_file.exists():
            urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                task_file
                )

        base_options = python.BaseOptions(model_asset_path=str(task_file))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2
        )

        self.hand_landmarker = vision.HandLandmarker.create_from_options(options)
    
    
    #def process_landmarks(self, hand_landmarks, handedness):
        #landmarks = []
        #for hand_num in range(len(self.results.hand_landmarks)):
            #current_hand = hand_landmarks[hand_num]
            #current_handedness = handedness[hand_num][0]
            #landmark_list = []
            #for id, landmark in enumerate(current_hand):
                #center_x, center_y, center_z = float(landmark.x), float(landmark.y), float(landmark.z)
                #landmark_list.append([id, center_x, center_y, center_z])
            
            #landmarks.append((current_handedness.category_name, landmark_list))


    def preprocess_image(self, webcam_frame):
        img_rgb = cv2.cvtColor(webcam_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=img_rgb
        )
        return mp_image

    def draw_overlay_hands(self, webcam_frame, overlay_mask, text_lr:tuple[str, str]=("Left", "Right")):
        """
        text: defaults to drawing "Left" at Left wrist coord, "Right" at Right wrist coord
        """
        height, width, n_channels = webcam_frame.shape
        for idx in range(len(self.current_hand_landmarks)):
            # Retrieve the wrist coordinate:
            wrist_coordinates = self.current_hand_landmarks[idx][0]
            wrist_x = wrist_coordinates.x
            wrist_y = wrist_coordinates.y
            #wrist_z = wrist_coordinates.z
            wrist_handedness = self.current_handedness[idx][0].category_name


            #curr_hand_landmarks = hand_landmarks[idx]
            #curr_handedness = handedness[idx]

            # =====================================================================
            # Draw handedness (left or right hand) on the image.
            # =====================================================================
            # Draw text at the top left corner of the detected hand's bounding box.
            #x_coordinates = [landmark.x for landmark in hand_landmarks]
            #y_coordinates = [landmark.y for landmark in hand_landmarks]
            #text_x = width - int(max(x_coordinates) * width)
            #text_y = int(min(y_coordinates) * height)
            #hand_label = f"{handedness[0].category_name}"

            # Draw text at the wrist coordinate (index 0)
            hand_label = wrist_handedness
            if hand_label == "Left":
                text = text_lr[0]
            else:
                text = text_lr[1]

            text_x = width - int(wrist_x * width)
            text_y = int(wrist_y * height)

            # =====================================================================
            font_scale = 1  # negative value to flip text vertically
            font_thickness = 1
            font_color = (88, 205, 54) # vibrant green

            cv2.putText(overlay_mask, text,
                        (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                        font_scale, font_color, font_thickness, cv2.LINE_AA)

            # =====================================================================
            # =====================================================================


            # Draw the hand landmarks.
            vision.drawing_utils.draw_landmarks(
                webcam_frame,
                self.current_hand_landmarks[idx],
                vision.HandLandmarksConnections.HAND_CONNECTIONS,
                vision.drawing_styles.get_default_hand_landmarks_style(),
                vision.drawing_styles.get_default_hand_connections_style())

        #cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)

    def __call__(self, webcam_frame):
        """
        returns `(handedness, hand_landmarks)`
        """

        results = self.hand_landmarker.detect(
            self.preprocess_image(webcam_frame)
            )

        self.current_handedness = results.handedness
        self.current_hand_landmarks = results.hand_landmarks

        return self.current_handedness, self.current_hand_landmarks


class GestureDetection:
    def __init__(self, src_directory = "gesture_detection/custom_dataset"):
        pass

        self.src_directory = Path("custom_dataset")
        label_map_path = self.src_directory / "label_map.csv"
        label_map = pd.read_csv(label_map_path)
        label_to_name = dict(zip(label_map["label"].astype(int), label_map["gesture_name"]))
        name_to_label = dict(zip(label_map["gesture_name"], label_map["label"].astype(int)))

    def mediapipe_to_tensor(self, handedness, hand_landmarks, isolated_hand=None):
        """
        `isolated_hand=None|"Left"|"Right"`: flips to / keeps only specified hand if specified
        """
        hand_coords_dict = self.create_hands_dict(handedness, hand_landmarks)
        left_hand_coords = None
        right_hand_coords = None
        if "Right" in hand_coords_dict.keys():
            right_hand_coords = hand_coords_dict["Right"]
        if "Left" in hand_coords_dict.keys():
            left_hand_coords = hand_coords_dict["Left"]

        tensor = self.landmarks_to_tensor(left_hand_coords, right_hand_coords, isolated_hand)
        return self.normalize_to_wrist(tensor)

    def expand_one_hand_to_two_hands(self, tensor, isolated_hand="Left"):
        """
        given a tensor of a single hand, represent it as full tensor with zeros for missing hand
        """
        if isolated_hand == "Left":
            return torch.cat((tensor, torch.zeros(63)))
        elif isolated_hand == "Right":
            return torch.cat((torch.zeros(63), tensor))


    def landmarks_to_tensor(self, left_hand_coords: list[(int, float, float, float)] = None, right_hand_coords: list[(int, float, float, float)] = None, isolated_hand=None) -> torch.Tensor:
        """
        index 0-62: left hand coordinates (x0, y0, z0, x1, y1, z1, ..., )
        index 63-126: right hand coordinates (x21, y21, z21, x22, y22, z22, ...,)

        `isolated_hand=None|"Left"|"Right"`: flips to / keeps only specified hand if specified
        """
        output_tensor = torch.zeros(63*2, dtype=torch.float32)
        lh_coords = []
        if left_hand_coords is not None:
            for i,x,y,z in left_hand_coords:
                lh_coords.extend([x,y,z])
            output_tensor[:63] += torch.tensor(lh_coords, dtype=torch.float32)

        rh_coords = []
        if right_hand_coords is not None:
            for i,x,y,z in right_hand_coords:
                rh_coords.extend([x,y,z])
            output_tensor[63:] += torch.tensor(rh_coords, dtype=torch.float32)

        if isolated_hand is None:
            return output_tensor

        elif isolated_hand == "Left":
            if output_tensor[:63].sum() > 0:
                return output_tensor[:63]
            else:
                # flip right hand to be left, assumes that coordinates are normalized to 0-1
                return torch.tensor([0,1,0]).repeat(21) - output_tensor[63:]

        elif isolated_hand == "Right":
            if output_tensor[63:].sum() > 0:
                return output_tensor[63:]
            else:
                # flip left hand to right, assumes that coordinates are normalized to 0-1
                return torch.tensor([0,1,0]).repeat(21) - output_tensor[:63]
        else:
            return output_tensor


    def normalize_to_wrist(self, landmark_tensor):
        """
        works for both single hand and both hand tensors
        """
        # subtract wrist (x,y,z) from all coordinates except wrist
        # Left hand block: indices 0..62 (21 points * 3)
        if landmark_tensor.numel() >= 63:
            landmark_tensor[3:63] -= landmark_tensor[0:3].repeat(21 - 1)

        # Right hand block: indices 63..125 (21 points * 3)
        if landmark_tensor.numel() >= 126:
            landmark_tensor[66:126] -= landmark_tensor[63:66].repeat(21 - 1)

        return landmark_tensor

    def create_hands_dict(self, handedness, hand_landmarks):
        """
        Returns the following dict:
        ```
        {
            "Left": [(0, x,y,z), (1,x,y,z) ...],
            "Right": [(0,x,y,z), (1,x,y,z), ...]
        }
        ```
        """
        hands_dict = {}
        for hand_num in range(len(hand_landmarks)):
            current_hand_landmarks = hand_landmarks[hand_num]
            hand_name = handedness[hand_num][0].category_name
            coords = []
            for id, landmark in enumerate(current_hand_landmarks):
                center_x, center_y, center_z = float(landmark.x), float(landmark.y), float(landmark.z)
                coords.append([id, center_x, center_y, center_z])

            hands_dict[hand_name] = coords
        return hands_dict

    def __call__(self, hands_dict):

        pass




class RecipeInterface:
    def __init__(self):
        self.recipe = {}



class Frontend:
    def __init__(self):
        self.landmarker = MediaPipeLandmarker()
        self.webcam = cv2.VideoCapture(0)
        self.gesture_detection = GestureDetection()

    def start(self):
        pass

    def run_webcam(self):
        frame_counter = 0
        print("Press Q to exit")
        while True:
            success, webcam_frame = self.webcam.read()
            key = cv2.waitKey(1) & 0xFF

            height, width, n_channels = webcam_frame.shape

            handedness, hand_landmarks = self.landmarker(webcam_frame)
            overlay_mask = np.zeros((height, width, n_channels), dtype="uint8")

            hand_coords_dict = self.gesture_detection.create_hands_dict(handedness, hand_landmarks)

            ####################################
            # Drawing onto webcam

            self.landmarker.draw_overlay_hands(webcam_frame, overlay_mask)

            # ...



            webcam_frame = cv2.flip(webcam_frame, 1)
            webcam_frame = cv2.add(webcam_frame, overlay_mask)
            # draw to screen
            cv2.imshow("Image", webcam_frame)

            ####################################



            if frame_counter % 25 == 0 and len(hand_landmarks) != 0:
                output_str = ""
                # note: output is horizontally flipped, so x=1-x, y=y
                #       is to keep top left as the origin (0,0)
                # 
                #       (1-x)*width, (y)*height is pixel coordinate
                for hand, coordinate in zip(handedness, hand_landmarks):
                    output_str += f"{hand[0].category_name}: x={1-coordinate[0].x}, y={coordinate[0].y}, "

                print(f"wrist coordinates: {output_str}")
                frame_counter = 0

            frame_counter += 1

            if key == ord("q"):
                break





if __name__ == "__main__":
    frontend = Frontend()
    frontend.run_webcam()
