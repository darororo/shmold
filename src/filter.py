import cv2
import os
from cv2.typing import MatLike
import numpy as np
import random
import shutil

def add_hsv(img: MatLike, hue=0, sat=0, val=0):
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
        img = add_hsv(img, hue, sat, val)
        
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

def add_rgb(img: MatLike, red=0, green=0, blue=0):
    img_int16 = img.astype(np.int16)
    b, g, r = cv2.split(img_int16)

    b = np.clip(b + blue, 0, 255)
    g = np.clip(g + green, 0, 255)
    r = np.clip(r + red, 0, 255)

    bgr_edited = cv2.merge([b, g, r]).astype(np.uint8)
    return bgr_edited


def change_bgr_and_save(data_dir: str, output_dir: str, red=0, green=0, blue=0):
    # Create folders
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    image_dir = os.path.join(data_dir, 'images')
    label_dir = os.path.join(data_dir, 'labels')

    out_img_dir = os.path.join(output_dir, 'images')
    out_label_dir = os.path.join(output_dir, 'labels')

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)


    suffix = 'rgb'
    for filename in os.listdir(image_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue

        img_path = os.path.join(image_dir, filename)

        img = cv2.imread(img_path)
        img = add_rgb(img, red, green, blue)

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

def sharpen(img: MatLike):
    # Define a sharpening kernel
    kernel = np.array([[-1, -1, -1],
                    [-1,  9, -1],
                    [-1, -1, -1]])

    # Apply the kernel to the image
    sharpened = cv2.filter2D(img, -1, kernel)
    
    return sharpened
    
def sharpen_and_save(data_dir: str, output_dir: str):
    # Create folders
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    image_dir = os.path.join(data_dir, 'images')
    label_dir = os.path.join(data_dir, 'labels')

    out_img_dir = os.path.join(output_dir, 'images')
    out_label_dir = os.path.join(output_dir, 'labels')

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)


    suffix = 'sharp'
    for filename in os.listdir(image_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue

        img_path = os.path.join(image_dir, filename)

        img = cv2.imread(img_path)
        img = sharpen(img)

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
            
def equalize_histogram(img: MatLike):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # Apply to grayscale image
    clahe_result = clahe.apply(gray)
    
    return clahe_result


def equalize_histogram_and_save(data_dir: str, output_dir: str):
     # Create folders
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    image_dir = os.path.join(data_dir, 'images')
    label_dir = os.path.join(data_dir, 'labels')

    out_img_dir = os.path.join(output_dir, 'images')
    out_label_dir = os.path.join(output_dir, 'labels')

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)


    suffix = 'eqhist'
    for filename in os.listdir(image_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue

        img_path = os.path.join(image_dir, filename)

        img = cv2.imread(img_path)
        img = equalize_histogram(img)

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