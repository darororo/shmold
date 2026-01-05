import cv2
import os
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

print("Augmentation complete. Images and labels saved in:", augmented_path)        
