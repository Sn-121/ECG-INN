import cv2
import numpy as np

img = cv2.imread("E:/2037.png",cv2.IMREAD_UNCHANGED)

img = cv2.resize(img,dsize=(600,400))

cv2.imwrite("E:/20377.png",img)