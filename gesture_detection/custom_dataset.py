from pathlib import Path
#from dataloader import ImageToTensorPreprocessor
import pandas as pd
import cv2
import numpy as np
from random import random
from pathlib import Path
from shutil import copy
from pathlib import Path
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, DataLoader
from typing import Union

import time
import sys
sys.path.insert(0, '..')
import frontend

from collections import defaultdict


class CustomDatasetCreator:
    def __init__(self, src_directory="custom_dataset"):
        self.landmarker = frontend.MediaPipeLandmarker()
        self.gesture_detection = frontend.GestureDetection()
        self.webcam = cv2.VideoCapture(0)

        self.src_directory = Path(src_directory)
        self.src_directory.mkdir(parents=True, exist_ok=True)
        
        self.label_map_path = self.src_directory / "label_map.csv"

        if not self.label_map_path.exists():
            default_label_map = {"label": [0, 1, 2], "gesture_name": ["gesture_0", "gesture_1", "gesture_2"]}
            pd.DataFrame(default_label_map).to_csv(self.label_map_path, index=False)

        label_map = pd.read_csv(self.label_map_path)
        self.label_to_name = dict(zip(label_map["label"].astype(int), label_map["gesture_name"]))
        self.name_to_label = dict(zip(label_map["gesture_name"], label_map["label"].astype(int)))


    def write_label_map(self):
        output_dict = defaultdict(list)
        for name,label in self.name_to_label.items():
            output_dict["gesture_name"].append(name)
            output_dict["label"].append(label)
        pd.DataFrame(output_dict).to_csv(self.label_map_path)

    def __call__(self, gesture_name="palm_up"):
        
        gesture_dir = self.src_directory / gesture_name
        gesture_dir.mkdir(parents=True, exist_ok=True)
        
        # capture every two seconds
        t = 2

        idx = 0
        records = []

        
        if gesture_name not in self.name_to_label:
            label = max(self.name_to_label.values()) + 1
            self.name_to_label[gesture_name] = label
            self.write_label_map()
        else:
            label = self.name_to_label[gesture_name]

        csv_path = self.src_directory / f"{label}_{gesture_name}_records.csv"
        if csv_path.exists():
            try:
                existing_records = pd.read_csv(csv_path)
                idx = existing_records["index"].max() + 1
            except:
                idx = 0

        
        last_capture_time = time.time()
        
        has_pressed_start = False
        print("press W to start")
        while True:
            ret, webcam_frame = self.webcam.read()
            if not ret:
                break

            height, width, n_channels = webcam_frame.shape
            handedness, hand_landmarks = self.landmarker(webcam_frame)
            overlay_mask = np.zeros((height, width, n_channels), dtype="uint8")
            self.landmarker.draw_overlay_hands(webcam_frame, overlay_mask)
            webcam_frame = cv2.flip(webcam_frame, 1)
            webcam_frame = cv2.add(webcam_frame, overlay_mask)


            if cv2.waitKey(1) & 0xFF == ord('w'):
                has_pressed_start = True

            if not has_pressed_start:
                cv2.imshow("Press 'W' to start", webcam_frame)
                continue
            else:
                cv2.imshow("Press 'Q' to exit", webcam_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            current_time = time.time()
            if current_time - last_capture_time >= t:
                
                if hand_landmarks is not None:
                    tensor = self.gesture_detection.mediapipe_to_tensor(handedness, hand_landmarks)
                    
                    tensor_path = gesture_dir / f"{idx}.pt"
                    torch.save(tensor, tensor_path)
                    
                    records.append({"index": idx, "label": label, "gesture_name": gesture_name})
                    idx += 1
                    print(f"captured sample with idx={idx}")
                
                last_capture_time = current_time
            
        
        cv2.destroyAllWindows()
        
        if records:
            df = pd.DataFrame(records)
            if csv_path.exists():
                df.to_csv(csv_path, mode='a', header=False, index=False)
            else:
                df.to_csv(csv_path, index=False)
        print(f"data added for {gesture_name}: {len(records)} records")
        print("all done")



class CustomDatasetProcessor:
    def __init__(self, src_directory="custom_dataset", train_val_split=0.8) -> None:
        self.train_val_split = train_val_split

        self.train_index = 0
        self.val_index = 0
        self.test_index = 0

        self.src_directory = Path(src_directory)
        self.label_map_path = self.src_directory / "label_map.csv"
        
        
        self.label_map = pd.read_csv(self.label_map_path)
        self.label_to_name = dict(zip(self.label_map["label"].astype(int), self.label_map["gesture_name"]))
        self.name_to_label = dict(zip(self.label_map["gesture_name"], self.label_map["label"].astype(int)))

        self.num_classes = len(self.label_to_name)

        self.src_train_val_dir = self.src_directory / "train"

        self.tgt_directory = self.src_directory.parent / f"{self.src_directory.name}_processed"

        self.tgt_train = self.tgt_directory / "train"
        self.tgt_train.mkdir(parents=True, exist_ok=True)

        self.tgt_val = self.tgt_directory / "val"
        self.tgt_val.mkdir(parents=True, exist_ok=True)

        self.dataset = {"index": [], "partition": [], "label": [], "position": []}

        self._process_dataset()
        print('processed dataset!')

        df = pd.DataFrame(self.dataset)
        df.to_csv(self.tgt_directory / "gestures.csv", index=False)
    
    def _add_record(self, index: int, partition: str, label: int, position: str):
        self.dataset["index"].append(index)
        self.dataset["partition"].append(partition)
        self.dataset["label"].append(label)
        self.dataset["position"].append(position)

    def _process_dataset(self):
        for _, row in self.label_map.iterrows():
            gesture_name = row["gesture_name"]
            label = int(row["label"])
            
            gesture_dir = self.src_directory / gesture_name
            if not gesture_dir.exists():
                continue
            
            for example in gesture_dir.iterdir():
                if example.suffix == ".pt":
                    random_number = random()
                    if random_number > self.train_val_split:
                        copy(Path.cwd() / example, self.tgt_val / f"{self.val_index}.pt")
                        self._add_record(self.val_index, "val", label, gesture_name)
                        self.val_index += 1
                    else:
                        copy(Path.cwd() / example, self.tgt_train / f"{self.train_index}.pt")
                        self._add_record(self.train_index, "train", label, gesture_name)
                        self.train_index += 1
            

class ImageDataset(Dataset):
    def __init__(
            self,
            directory: Union[str, Path],
            partition: str = "train",
            indices: list[int] = None,
        ):
            self.partition = partition
            if partition not in ("train", "test", "val"):
                raise ValueError(f"Invalid partition specified - {partition}")
            self.directory = Path(directory)
            self.img_directory = self.directory / partition
            metadata = pd.read_csv(self.directory / "gestures.csv")
            #self.num_classes = metadata["label"].nunique()
            
            if indices is not None:
                # Use provided indices (for stratified split)
                self.metadata = metadata.iloc[indices].reset_index(drop=True)
            else:
                # Use partition column
                self.metadata = metadata[metadata["partition"] == partition]
                self.metadata.reset_index(drop=True, inplace=True)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index) -> tuple[torch.Tensor, int]:
        row = self.metadata.iloc[index]
        label = row["label"]
        tensor_idx = row["index"]
        tensor_path = str(self.img_directory / f"{tensor_idx}.pt")
        
        landmark_tensor = torch.load(tensor_path)
        
        if landmark_tensor is None:
            raise RuntimeError(f"Failed to load image at {tensor_path}. File may be missing or corrupted.")

        return landmark_tensor, label


def get_dataloader(
    dataset_name: str = "custom_dataset",
    partition: str = "train",
    batch_size: int = 1,
    num_workers = 0,
    shuffle: bool = None,
    train_val_split: float = 0.8,
    random_state: int = 42,
) -> DataLoader:
    """
    dataset: "custom_dataset"|"hands_dataset"
    partition: "train" | "val"

    Performs stratified train/val split based on labels.
    """

    assert (partition == "train") or (partition == "val")

    directory = f"{dataset_name}_processed"
    metadata = pd.read_csv(Path(directory) / "gestures.csv")
    train_val_metadata = metadata[metadata["partition"] != "test"]
    
    if len(train_val_metadata) > 0:
        labels = train_val_metadata["label"].values
        indices = train_val_metadata.index.values
        
        sss = StratifiedShuffleSplit(n_splits=1, test_size=1-train_val_split, random_state=random_state)
        train_idx, val_idx = next(sss.split(indices, labels))
        
        if partition == "train":
            selected_indices = indices[train_idx]
        else:
            selected_indices = indices[val_idx]
    else:
        selected_indices = train_val_metadata.index.values
    
    dataset = ImageDataset(
        directory=directory,
        partition="train",
        indices=list(selected_indices)
    )

    loader = DataLoader(
        dataset = dataset,
        batch_size=batch_size, 
        num_workers=num_workers, 
        shuffle=shuffle
        )
        
    return loader



if __name__ == "__main__":
    #data_creator = CustomDatasetCreator()
    #data_creator(gesture_name = "palm_up")
    #data_creator(gesture_name = "palm_down")

    processor = CustomDatasetProcessor()