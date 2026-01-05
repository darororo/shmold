import cv2
import src.filter as filter

def main():
    print("Hello from shmold!")

    filter.sharpen_and_eq_hist()

if __name__ == "__main__":
    main()