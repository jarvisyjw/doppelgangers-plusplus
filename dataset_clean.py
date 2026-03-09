import numpy as np
from PIL import Image
import os
from tqdm import tqdm  # specific library for progress bars
from collections import defaultdict

def is_valid_image(path):
    """
    Tries to open and verify an image file. 
    Returns True if the image is valid, False if corrupted or missing.
    """
    try:
        if not os.path.exists(path):
            return False
            
        with Image.open(path) as img:
            # img.verify() # Check for file corruption without decoding the whole image
            img.load()  # Force loading the image to catch more subtle corruption issues
        return True
    except (IOError, SyntaxError, OSError):
        return False

def filter_npy_dataset(data_root, input_npy_path, output_npy_path):
    # 1. Load the original data
    print(f"Loading data from {input_npy_path}...")
    data = np.load(input_npy_path, allow_pickle=True)
    
    print(f"Original dataset shape: {data.shape}")
    
    valid_pairs = []
    corrupted_count = 0
    
    # 2. Iterate with a progress bar
    print("Checking images...")
    for pair in tqdm(data):
        # Assuming pair is [path1, path2]
        path1, path2 = pair[0], pair[1], 
        path1 = os.path.join(data_root, path1)
        path2 = os.path.join(data_root, path2)
        
        # Check both images in the pair
        if is_valid_image(path1) and is_valid_image(path2):
            valid_pairs.append(pair)
        else:
            corrupted_count += 1
            # Optional: Print which file was bad
            # print(f"Skipping pair due to error: {path1} or {path2}")

    # 3. Convert back to numpy array and save
    valid_array = np.array(valid_pairs)
    
    print("-" * 30)
    print(f"Processing complete.")
    print(f"Original samples: {len(data)}")
    print(f"Valid samples:    {len(valid_array)}")
    print(f"Removed samples:  {corrupted_count}")
    
    if corrupted_count > 0:
        print("Some pairs were removed due to missing or corrupted images.")
        np.save(output_npy_path, valid_array)
        print(f"Saved cleaned dataset to {output_npy_path}")
    else:
        print("No corrupted images found. No changes made to the dataset.")

if __name__ == "__main__":
    # CONFIGURATION
    DATA_ROOT = "data/doppelgangers/images/train_set_flip"  # Base directory for image paths in the dataset
    INPUT_FILE = "data/pairs_metadata/train_pairs_flip.npy"  # Replace with your file
    OUTPUT_FILE = "data/pairs_metadata/train_pairs_flip_cleaned.npy"
    
    filter_npy_dataset(DATA_ROOT, INPUT_FILE, OUTPUT_FILE)