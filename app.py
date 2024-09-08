import numpy as np
import cv2



img = cv2.imread('img dataset/0.png',cv2.IMREAD_GRAYSCALE)
img_array = np.array(img)

print(img_array)
print(img_array.shape)