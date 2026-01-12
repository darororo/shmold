import cv2
import matplotlib.pyplot as plt
import src.filter as filter

def main():

    data_dir = 'data/project-17-at-2026-01-12-12-18-01b29e6f'
    output_dir = 'data/output/hsv5'

    filter.change_hsv_and_save(data_dir, output_dir, -15, 5, 2)

    # img_path = 'data/project-17-at-2026-01-12-12-18-01b29e6f/images/008daf3a-Dropped_Image_124.png'

    # img = cv2.imread(img_path)
    # img = filter.hsv(img, -20, -5, 0)

    # cv2.imshow('change hsv', img)

    # while True:
    #     key = cv2.waitKey(1) & 0xFF
    #     if key == 27:  # ESC key
    #         break

    # cv2.destroyAllWindows()
    # filter.sharpen_and_eq_hist()

if __name__ == "__main__":
    main()