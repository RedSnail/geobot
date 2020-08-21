import cv2
import numpy as np

nimg = 3
imlist = []

colored = cv2.imread("2_full.jpg")[200:1400, 200:2260]

for i in range(nimg):
    img = cv2.imread(f"{i+1}_full.jpg", cv2.IMREAD_GRAYSCALE)
    cut = img[200:1400, 200:2260]
    # binary = cv2.adaptiveThreshold(cut, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)
    ret, binary = cv2.threshold(cut, 107, 255, cv2.THRESH_BINARY_INV)
    imlist.append(binary)


imdiff = []
for i in range(nimg-1):
    imlist[0][imlist[0] > imlist[i+1]] = imlist[i+1][imlist[0] > imlist[i+1]]
    imdiff.append(imlist[i+1] - imlist[0])

kernel = np.ones((7, 7), np.uint8)
erosion = cv2.erode(imdiff[1] & imdiff[0], kernel, iterations=2)
dilation = cv2.dilate(erosion, kernel, iterations=3)
gradient = cv2.morphologyEx(dilation, cv2.MORPH_GRADIENT, kernel)
contours, hierarchy = cv2.findContours(gradient, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(colored, contours, -1, (0, 255, 0), 5)
cv2.imwrite("detected.jpg", colored)

