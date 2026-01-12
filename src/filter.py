import cv2
import os
from cv2.typing import MatLike
import numpy as np
import random
import shutil

sharpen_prob = 0.4      # 40% chance to sharpen
hist_eq_prob = 0.3       # 30% chance to equalize

sharpen_kernel = np.array([
    [0, -1,  0],
    [-1, 5, -1],
    [0, -1,  0]
])

# Path
data_image_path = 'data/v2/train/images'
data_labels_path = 'data/v2/train/labels'

augmented_path = 'data/v2/augmented'
aug_images_path = os.path.join(augmented_path, 'images')
aug_labels_path = os.path.join(augmented_path, 'labels')

# Create folders
os.makedirs(aug_images_path, exist_ok=True)
os.makedirs(aug_labels_path, exist_ok=True)

def sharpen_and_eq_hist():
    for filename in os.listdir(data_image_path):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue
        
        img_path = os.path.join(data_image_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        
        if random.random() < sharpen_prob:
            img = cv2.filter2D(img, -1, sharpen_kernel)
        
        if random.random() < hist_eq_prob:
            img = cv2.equalizeHist(img)
        
        # Get file name and extension separately
        name, ext = os.path.splitext(filename)
        
        # Save augmented images    
        save_name = f'{name}_sharp_eq_hist{ext}'
        save_path = os.path.join(aug_images_path, save_name)
        cv2.imwrite(save_path, img)
        
        
        # Copy all labels from OG file
        label_file = name + '.txt'
        src_label_path = os.path.join(data_labels_path, label_file)
        dst_label_path = os.path.join(aug_labels_path, f'{name}_sharp_eq_hist.txt')
        if os.path.exists(src_label_path):
            shutil.copy(src_label_path, dst_label_path)

def hsv(img: MatLike, hue=0, sat=0, val=0):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    h, s, v = cv2.split(hsv)

    h = (h + hue) % 180        # wrap hue
    s = np.clip(s + sat, 0, 255)
    v = np.clip(v + val, 0, 255)

    hsv_edited = cv2.merge([h, s, v]).astype(np.uint8)
    img_edited = cv2.cvtColor(hsv_edited, cv2.COLOR_HSV2BGR)
    return img_edited


def change_hsv_and_save(data_dir: str, output_dir: str, hue=0, sat=0, val=0):
    # Create folders
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    image_dir = os.path.join(data_dir, 'images')
    label_dir = os.path.join(data_dir, 'labels')

    out_img_dir = os.path.join(output_dir, 'images')
    out_label_dir = os.path.join(output_dir, 'labels')

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)


    suffix = 'hsv'
    for filename in os.listdir(image_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue

        img_path = os.path.join(image_dir, filename)

        img = cv2.imread(img_path)
        img = hsv(img, hue, sat, val)
        
        # Get file name and extension separately
        name, ext = os.path.splitext(filename)

        # Save augmented images
        save_name = f'{name}_{suffix}{ext}'
        save_path = os.path.join(out_img_dir, save_name)
        _ = cv2.imwrite(save_path, img)


        # Copy all labels from OG file
        label_file = name + '.txt'
        src_label_path = os.path.join(label_dir, label_file)
        dst_label_path = os.path.join(out_label_dir, f'{name}_{suffix}.txt')
        if os.path.exists(src_label_path):
            _ = shutil.copy(src_label_path, dst_label_path)

