import cv2
import numpy as np
import matplotlib.pyplot as plt


def hough(image):
    theta = np.deg2rad(np.arange(-90.0, 90.0))
    diag = int(np.ceil(np.sqrt(image.shape[0]*image.shape[0] + image.shape[1]*image.shape[1])))
    ps = np.linspace(-diag, diag, diag*2.0)

    accumulator = np.zeros((2*diag, len(theta)), dtype=np.uint64)
    y_idxs, x_idxs = np.nonzero(image)

    cos = np.cos(theta)
    sin = np.sin(theta)

    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        for j in range(len(theta)):
            p = int(diag) + int(round(x*cos[j] + y*sin[j]))
            accumulator[p, j] += 1

    return accumulator, theta, ps


img = cv2.imread('2.png', 0)
img1 = cv2.imread('2.png')
edges = cv2.Canny(img, 50, 300)

accumulator, thetas, ps = hough(edges)

print(len(thetas))
print(len(ps))

print(accumulator[1087, 85])

for i in range(accumulator.shape[0]):
    for j in range(accumulator.shape[1]):
        if accumulator[i][j] > 150:
            a = np.cos(thetas[j])
            b = np.sin(thetas[j])
            x0 = a * ps[i]
            y0 = b * ps[i]
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * a)

            cv2.line(img1, (x1, y1), (x2, y2), [50, 50, 200], 2)

cv2.imshow('lines', img1)
plt.imshow(accumulator)
plt.show()
cv2.waitKey(0)