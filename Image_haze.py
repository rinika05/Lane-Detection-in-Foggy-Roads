
import cv2
import numpy as np
from defogging import Defog



def canny_edge(a):
    # converting to grayscale
    gray_lane = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)

    # to make edges smooth add smoothening effect through Gaussian filter
    # Increase the mask area to make only road white line prominent
    # refer: https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
    lane_blur = cv2.GaussianBlur(gray_lane, (11, 11), 2)

    # use Canny edge detection method to define lane edges (intensity transition)
    canny_lane = cv2.Canny(lane_blur, 50, 200)
    return canny_lane


def average_slope_intercept(a, lines):
    left = []  # contains coordinates of the line on left
    right = []  # contains coordinates of the line on right
    for l in lines:
        x1, y1, x2, y2 = l.reshape(4)
        param = np.polyfit((x1, x2), (y1, y2), 1)  # slope & y-incercept
        slope = param[0]
        intercept = param[1]
        if slope < 0:
            left.append((slope, intercept))
        else:
            right.append((slope, intercept))
        left_average = np.average(left, axis=None, weights=None, returned=False)
        right_average = np.average(right, axis=None, weights=None, returned=False)


def region_of_interest(a):
    # Finding region of interest
    height = a.shape[0]
    pol = np.array([
        [(0, height), (300, height), (250, 200)]
    ])
    mask = np.zeros_like(a)
    cv2.fillPoly(mask, pol, 255)
    m_img = cv2.bitwise_and(a, mask)
    return m_img


def lines_in_image(a, lines):
    line_img = np.zeros_like(a)
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l.reshape(4)       #Reshaping 2D array to 1D array with 4 elements
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0),3)
    return line_img


df = Defog()
df.read_img('1.jpg')
df.defog()
df.save_img('test.jpg')
a = cv2.imread("test.jpg")
canny_lane = canny_edge(a)
reg_image = region_of_interest(canny_lane)
lines = cv2.HoughLinesP(reg_image, 4, np.pi / 180, 80, np.array([]), minLineLength=10, maxLineGap=5)
    # np.pi * 1deg/180 = 1rad
l_img = lines_in_image(a, lines)
    # Blending the lines in original image
final_img = cv2.addWeighted(a, 0.8, l_img, 1, 1)
cv2.imshow("original", cv2.imread("1.jpg"))
cv2.imshow("dehazed", a)
cv2.imshow("result", canny_lane)
cv2.waitKey(0)
