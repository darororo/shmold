import cv2
import src.filter as filter

def main():
    print("Hello from shmold!")

    img = cv2.imread('image/mold/c.jpeg', cv2.IMREAD_GRAYSCALE)
    img = filter.apply_histogram(img)
    cv2.imshow('display-1', img)




if __name__ == "__main__":
    main()