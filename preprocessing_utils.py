import cv2
import numpy as np


def _center_crop(img, dim):
	width, height = img.shape[1], img.shape[0]
	crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
	crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0]
	mid_x, mid_y = int(width/2), int(height/2)
	cw2, ch2 = int(crop_width/2), int(crop_height/2)
	crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
	return crop_img


def _scale_image(img, factor=1):
	return cv2.resize(img,(int(img.shape[1]*factor), int(img.shape[0]*factor)))



def _get_random_crop(image, crop_height, crop_width):
	x, y =0, 0

	max_x = image.shape[1] - crop_width
	max_y = image.shape[0] - crop_height
	if max_x == 0 and max_y!=0:
		x = 0
		y = np.random.randint(0, max_y)

	if max_y == 0 and max_x!=0:
		y = 0
		x = np.random.randint(0, max_x)

	if max_x!=0 and max_y!=0:
		x = np.random.randint(0, max_x)
		y = np.random.randint(0, max_y)

	crop = image[y: y + crop_height, x: x + crop_width]

	return crop