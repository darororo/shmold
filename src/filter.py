import cv2
import numpy as np


def apply_histogram(img_mat):
	res = cv2.equalizeHist(img_mat, None)

	return res
