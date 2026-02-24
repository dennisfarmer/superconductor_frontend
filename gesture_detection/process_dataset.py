#!/usr/bin/env python
#
# Dennis Farmer
#
# Used also for 
# Michigan Data Science Team Winter 2026:
# American Sign Language Translation

import shutil
from pathlib import Path
from dataloader import HandsDatasetProcessor

def process_dataset(dataset_name: str):
    processed_dir = Path(dataset_name).parent / f"{dataset_name}_processed"
    if processed_dir.exists():
        shutil.rmtree(processed_dir)
    
    print(f"Processing {dataset_name} with landmark filtering...")
    processor = HandsDatasetProcessor(
        src_directory=dataset_name,
        filter_to_landmarkable=True,
    )
    print(f"Dataset saved to {processed_dir}")

if __name__ == "__main__":
    process_dataset("hands_dataset")
