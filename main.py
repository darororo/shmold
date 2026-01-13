import cv2
import matplotlib.pyplot as plt
import src.filter as filter

def main():

    data_dir = '/home/eva/Documents/ITC/I5/ImageProcessing/mold_detection/shmold/data/polygon/train'

    rgb_output_dir = '/home/eva/Documents/ITC/I5/ImageProcessing/mold_detection/shmold/data/rgb/train'
    hsv_output_dir = '/home/eva/Documents/ITC/I5/ImageProcessing/mold_detection/shmold/data/hsv/train'
    sharp_output_dir = '/home/eva/Documents/ITC/I5/ImageProcessing/mold_detection/shmold/data/sharp/train'

    # filter.change_hsv_and_save(data_dir, hsv_output_dir, -15, 5, 2)
    # filter.change_bgr_and_save(data_dir, rgb_output_dir, 0, 10, -10)
    filter.sharpen_and_save(data_dir, sharp_output_dir)

    img_path = '/home/eva/Documents/ITC/I5/ImageProcessing/mold_detection/shmold/data/polygon/train/images/f3615f65-Dropped_Image_50.png'

    img = cv2.imread(img_path)
    imgRGB = filter.add_rgb(img, 2, 20, 40)
    imgHSV = filter.add_hsv(img, -10, 20, 16)
    imgSharp = filter.sharpen(img)


    cv2.imshow('change rgb', imgRGB)
    cv2.imshow('change hsv', imgHSV)
    cv2.imshow('change sharp', imgSharp)



    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break

    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()