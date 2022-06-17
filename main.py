# OpenCV-Python is a library of Python bindings designed to solve computer vision problems.
# cv2 is an open CV import format

import cv2
import numpy as np


def canny_edge(a):
    # converting to grayscale
    gray_lane = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)

    # to make edges smooth add smoothening effect through Gaussian filter
    # Increase the mask area to make only road white line prominent
    # refer: https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
    lane_blur = cv2.GaussianBlur(gray_lane, (11, 11), 1)

    # use Canny edge detection method to define lane edges (intensity transition)
    canny_lane = cv2.Canny(lane_blur, 20, 100)
    return canny_lane


def coord(a, l_param):
    slope, intercept = l_param
    y1 = a.shape[0]
    x1 = int(y1 - intercept) / slope
    y2 = int(y1 * (4 / 5))
    x2 = int(y2 - intercept) / slope
    return np.array([x1, y1, x2, y2])


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
        left_line = coord(a, left_average)
        right_line = coord(a, right_average)
        return np.array([left_line, right_line])


def region_of_interest(a):
    # Finding region of interest
    height = a.shape[0]
    pol = np.array([
        [(180, height), (1100, height), (550, 250)]
    ])
    mask = np.zeros_like(a)
    cv2.fillPoly(mask, pol, 255)
    m_img = cv2.bitwise_and(a, mask)
    return m_img


def lines_in_image(a, lines):
    line_img = np.zeros_like(a)
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l.reshape(4)
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 7)
    return line_img


a = cv2.VideoCapture("test2.mp4")
while (a.isOpened()):
    _, frame = a.read()
    canny_lane = canny_edge(frame)
    reg_image = region_of_interest(canny_lane)
    lines = cv2.HoughLinesP(reg_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    # np.pi * 1deg/180 = 1rad
    l_img = lines_in_image(frame, lines)
    # Blending the lines in original image
    final_img = cv2.addWeighted(frame, 0.8, l_img, 1, 1)
    cv2.imshow("result", final_img)  # displaying the image
    if cv2.waitKey(1) == ord('a'):
        break
a.release()
