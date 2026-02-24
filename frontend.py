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
from gesture_detection.model import PalmModel
from gesture_detection.dataloader import get_dataloader


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
    def __init__(self, model_name = "palm_hold_release"):
        """
        `model_name` can be either `"palm_up_down"` or `"palm_hold_release"`
        """
        self.model_name = model_name
        self.model_initialized = False
        self.device = None

    def initialize_model(self):
        self.src_directory = Path("gesture_detection")
        label_map_path = self.src_directory / self.model_name / "label_map.csv"
        label_map = pd.read_csv(label_map_path)
        self.label_to_name = dict(zip(label_map["label"].astype(int), label_map["gesture_name"]))
        name_to_label = dict(zip(label_map["gesture_name"], label_map["label"].astype(int)))
        self.num_classes = label_map["label"].nunique()

        if torch.cuda.is_available():
            print("Using GPU")
            self.device = torch.device("cuda")

        # MPS: Apple Silicon
        elif torch.backends.mps.is_available():
            print("Using MPS")
            self.device = torch.device("mps")

        # CPU: 
        else:
            print("Using CPU")
            self.device = torch.device("cpu")

        self.model = PalmModel(input_features = 63*2, num_classes=self.num_classes)
        self.model.load_state_dict(torch.load(f"gesture_detection/{self.model_name}_model.pth"))
        self.model.to(self.device)
        self.model.eval()
        self.model_initialized = True

    def __call__(self, tensor, isolated_hand="Left"):
        if not self.model_initialized:
            return None, None

        tensor = tensor.to(self.device)
        classification = self.model(tensor)
        predicted_label = torch.argmax(classification, dim=-1).item()
        predicted_gesture = self.label_to_name.get(predicted_label, f"Unknown ({predicted_label})")
        confidence = torch.softmax(classification, dim=-1).max().item() * 100

        return predicted_gesture, confidence

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




class RecipeInterface:
    def __init__(self, prompts, slider_up_gesture, slider_down_gesture, slider_neutral_gesture):
        self.recipe = {
            "Rock": 0.6,
            "Guitar": 0.8,
            "Jazz": 0.3
        }
        self.slider_up_gesture = slider_up_gesture
        self.slider_down_gesture = slider_down_gesture
        self.slider_neutral_gesture = slider_neutral_gesture

        self.bar_positions = {k: (None,None) for k in self.recipe.keys()}
        self.bar_colors = {
            # note: color channels are flipped RGB -> BGR
            "Rock": (78, 166, 216)[::-1],
            "Guitar": (250, 162, 75)[::-1],
            "Jazz": (150, 187, 136)[::-1]
        }
    
    def draw_bars(self, webcam_frame, overlay_mask):
        height, width, num_channels = webcam_frame.shape
        num_bars = max(1, len(self.recipe))
        margin = 20
        self.bar_width = max(20, int((width - (num_bars + 1) * margin) / num_bars))
        bar_height = int(height * 0.7)
        self.bar_top = int(height * 0.1)
        self.bar_bottom = self.bar_top + bar_height

        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        font_thickness = 1

        positions = {}
        for i, (label, value) in enumerate(self.recipe.items()):
            x1 = margin + i * (self.bar_width + margin)
            x2 = x1 + self.bar_width

            bar_color = self.bar_colors.get(label, (88, 205, 54))

            cv2.rectangle(overlay_mask, (x1, self.bar_top), (x2, self.bar_bottom), bar_color, 2)

            v = max(0.0, min(1.0, float(value)))
            fill_height = int(bar_height * v)
            fill_top = self.bar_bottom - fill_height
            if fill_height > 0:
                cv2.rectangle(overlay_mask, (x1 + 2, fill_top), (x2 - 2, self.bar_bottom - 2), bar_color, -1)

            text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
            text_x = x1 + (self.bar_width - text_size[0]) // 2
            text_y = self.bar_bottom + text_size[1] + 10
            cv2.putText(overlay_mask, label, (text_x, text_y), font, font_scale, bar_color, font_thickness, cv2.LINE_AA)

            center_x = x1 + self.bar_width // 2
            center_y = fill_top if fill_height > 0 else self.bar_bottom
            positions[label] = (center_x, center_y)

        self.bar_positions = positions
        return positions
    
    def adjust_recipe(self, prompt, pointer_y):
        bar_height = max(1, self.bar_bottom - self.bar_top)
        clamped_y = max(self.bar_top, min(self.bar_bottom, int(pointer_y)))
        proportion = (self.bar_bottom - clamped_y) / bar_height
        proportion = max(0.0, min(1.0, float(proportion)))
        if prompt in self.recipe:
            self.recipe[prompt] = proportion

    def update_positions(self, pointer_x, pointer_y, gesture):
        closest_bar = None
        min_distance = float('inf')
        
        for prompt, (bar_x, bar_y) in self.bar_positions.items():
            if bar_x is None:
                continue
            x_distance = abs(pointer_x - bar_x)
            if x_distance < self.bar_width and x_distance < min_distance:
                min_distance = x_distance
                closest_bar = prompt
        
        if closest_bar is not None:
            current_y = self.bar_positions[closest_bar][1]
            
            if gesture == self.slider_up_gesture and pointer_y < current_y:
                self.bar_positions[closest_bar] = (self.bar_positions[closest_bar][0], pointer_y)
                self.adjust_recipe(closest_bar, pointer_y)
            
            elif gesture == self.slider_down_gesture and pointer_y > current_y:
                self.bar_positions[closest_bar] = (self.bar_positions[closest_bar][0], pointer_y)
                self.adjust_recipe(closest_bar, pointer_y)
    
    def change_prompts(self, new_prompt_list):
        self.recipe = {p: 0 for p in new_prompt_list}



    



class Frontend:
    def __init__(self, prompts):
        self.model_name = "palm_hold_release"
        self.landmarker = MediaPipeLandmarker()
        self.webcam = cv2.VideoCapture(0)
        self.gesture_detection = GestureDetection(self.model_name)
        self.gesture_detection.initialize_model()

        self.recipe_interface = RecipeInterface(
            prompts=prompts,
            slider_up_gesture = "palm_release_up",
            slider_neutral_gesture = "palm_release_down",
            slider_down_gesture = "palm_hold"
        )

    def start(self):
        self.run_webcam()
        # multi threading for sending / receiving requests
        # ...

    def run_webcam(self):
        gesture_name = None
        middle_finger_x = None
        middle_finger_y = None

        print("Press Q to exit")
        while True:
            success, webcam_frame = self.webcam.read()
            key = cv2.waitKey(1) & 0xFF

            height, width, n_channels = webcam_frame.shape

            handedness, hand_landmarks = self.landmarker(webcam_frame)
            overlay_mask = np.zeros((height, width, n_channels), dtype="uint8")

            isolated_hand = "Left"

            if len(hand_landmarks) > 0:
                hand_tensor = self.gesture_detection.mediapipe_to_tensor(handedness, hand_landmarks, isolated_hand)
                hand_tensor = self.gesture_detection.expand_one_hand_to_two_hands(hand_tensor, isolated_hand)
                gesture_name, confidence = self.gesture_detection(hand_tensor, isolated_hand)

                # extract middle finger tip
                left_hand_landmarks = hand_landmarks[0]
                middle_finger = left_hand_landmarks[12]
                # Convert normalized coordinates (0-1) to pixel coordinates
                middle_finger_x = int((1 - middle_finger.x) * width)  # flip x for mirrored display
                middle_finger_y = int(middle_finger.y * height)

                self.recipe_interface.update_positions(
                    pointer_x=middle_finger_x,
                    pointer_y=middle_finger_y,
                    gesture=gesture_name
                )




            ####################################
            # Drawing onto webcam

            self.landmarker.draw_overlay_hands(
                webcam_frame,
                overlay_mask,
                text_lr = (f"{gesture_name} ({confidence:.1f}%)", "") if gesture_name is not None else ("no label", "")
            )

            self.recipe_interface.draw_bars(
                webcam_frame,
                overlay_mask
            )


            # ...
            webcam_frame = cv2.flip(webcam_frame, 1)
            webcam_frame = cv2.add(webcam_frame, overlay_mask)
            # draw to screen
            cv2.imshow("SuperConductor - Webcam View (Mediapipe)", webcam_frame)

            ####################################



            if key == ord("q"):
                break





if __name__ == "__main__":
    frontend = Frontend(prompts = ["Rock", "Guitar", "Jazz"])
    frontend.start()
