import cv2
import numpy as np

degs = 2.3
# read
img = cv2.imread('in.jpeg')

# rotate
rotation = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), degs, 1)
img = cv2.warpAffine(img, rotation, (img.shape[1], img.shape[0]))

# crop
img = img[265:888, 875:1005]
crop = img

# grayscale
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('gray.jpeg', img)

# edge detection
img = cv2.Canny(img, 100, 100, apertureSize=3)

# dilate and erode
kernel = np.ones((3, 3), np.uint8)
iters = 5
if iters:
    img = cv2.dilate(img, kernel, iterations=iters)
    img = cv2.erode(img, kernel, iterations=iters+1)

cv2.imwrite('edges.jpeg', img)

lines = cv2.HoughLines(img, 0.3, np.pi / 4, 75)
seglines = None
# lines = None
# seglines = cv2.HoughLinesP(edges, rho=1,
#                         theta=np.pi / 30,
#                         threshold=30,
#                         minLineLength=60,
#                         maxLineGap=50)

if seglines is not None:
    for line in seglines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # cv2.line(edges, (x1, y1), (x2, y2), (255, 255, 255), 1)

if lines is not None:
    for line in lines:
        for rho, theta in line:
            # ignore vertical lines
            if abs(theta - np.pi / 2) > 0.3:
                continue
            print(theta)
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(crop, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # cv2.line(crop, (x1, y1), (x2, y2), (255, 255, 255), 1)

    print("lines: " + str(len(lines)))

# checkpoint
cv2.imwrite('out.jpeg', crop)