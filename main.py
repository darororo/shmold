import cv2
import matplotlib.pyplot as plt
import src.filter as filter

def main():

    data_dir = 'data/project-17-at-2026-01-12-12-18-01b29e6f'


    rgb_output_dir = 'data/output/rgb5'
    hsv_output_dir = 'data/output/hsv5'

    # filter.change_hsv_and_save(data_dir, hsv_output_dir, -15, 5, 2)
    # filter.change_bgr_and_save(data_dir, rgb_output_dir, -50, 25, -20)

    img_path = 'data/project-17-at-2026-01-12-12-18-01b29e6f/images/008daf3a-Dropped_Image_124.png'

    img = cv2.imread(img_path)
    # img = filter.rgb(img, 2, 20, 40)
    img1 = filter.add_rgb(img, -10, 20, 16)

    img2 = filter.add_hsv(img, -12, 4, 0)


    cv2.imshow('change rgb', img1)
    cv2.imshow('change hsv', img2)


    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break

    cv2.destroyAllWindows()


    # filter.eq_hist()

if __name__ == "__main__":
    main()