from pathlib import Path
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

working_dir = Path.cwd()
filter_corrupted_data = True

if filter_corrupted_data:
    print("Removing corrupted images...")
    num_skipped = 0
    for folder_name in ["cat", "dog"]:
        folder_path = working_dir / "data" / "pet_images" / folder_name
        for fname in folder_path.glob("*.jpg"):
            fpath = folder_path / fname
            try:
                fobj = open(fpath, "rb")
                is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
            finally:
                fobj.close()

            if not is_jfif:
                num_skipped += 1
                os.remove(fpath)
    print(f"Deleted {num_skipped} corrupted images.")

# Generate a dataset (labels: 0=cat, 1=dog)
print("Generating train & test sets from repository...")
image_size = (180, 180)
train_data, test_data = tf.keras.utils.image_dataset_from_directory(
    "data/pet_images",
    validation_split=0.2,
    subset="both",
    seed=1234,
    image_size=image_size,
    batch_size=100,
)

# Apply data augmentation to the training dataset
print("Augment training set by adding random granularity (flipping & rotating images)...")
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ]
)
train_data = train_data.map(
    lambda img, label: (data_augmentation(img), label),
    num_parallel_calls=tf.data.AUTOTUNE,
)
