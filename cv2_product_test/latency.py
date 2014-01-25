# Copyright 2014 Jetperch LLC - See LICENSE file.
"""
Process touchscreen latency images into finger coordinate and screen coordinate.

The device's touchscreen must be illuminated white so that the finger shows
up black.  The device must display a magenta dot at the current touch location. 
"""

import os
import cv2
import numpy as np
import scipy.optimize


def _filter_image(img):
    kernel = np.ones((3,3),np.uint8)
    img = cv2.erode(img, kernel)
    return img


def separate_targets(img):
    """Separate the targets from the BGR image.
    
    The finger is mostly black and the touchscreen is magneta.
    
    :return: The tuple of images (finger, touchscreen).
    """
    
    img = 255 - img # Invert
    threshold = 96
    _, img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    # Colors stored as blue, green, red
    # finger is now white and touchscreen is now green
    touchscreen = img[:, :, 1] - img[:, :, 0]
    touchscreen = _filter_image(touchscreen)
    finger = img[:, :, 1] - touchscreen
    finger = _filter_image(finger)
    return (finger, touchscreen)


def finger_image_to_coordinates(img):
    """Find the most vertical point on the finger.
    
    :param img: The finger image.
    :return: The (x, y) point in screen coordinates (int).  If the finger 
        cannot be found, return None.
    """
    v = np.max(img, axis=1)
    try:
        y = np.where(v)[0][0]
        x = np.round(np.mean(np.where(img[y])[0]))
    except IndexError:
        return None
    return (int(x), int(y))


def touchscreen_image_to_coordinate(img):
    """Find the center of mass.
    
    :param img: The touchscreen image containing the processed circle.
    :return: The (x, y) point in screen coordinates (int).  If the touchscreen
        circle cannot be found, return None.
    """
    # http://docs.opencv.org/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#moments
    rv = cv2.moments(img, True)
    if rv['m00'] == 0:
        return None 
    x = rv['m10'] / rv['m00']
    y = rv['m01'] / rv['m00']
    x, y = np.round(x), np.round(y)
    return (int(x), int(y))


def process(img):
    """Process an image into finger and touchscreen coordinates.
    
    :param img: The image to process.
    :return: ((finger_x, finger_y), (touchscreen_x, touchscreeny)) as integer
        coordinates in the image.  If either finger or touchscreen is not 
        found, then the corresponding point will be None.
    """
    if np.prod(img.shape) == 0:
        return (None, None)
    finger, touchscreen = separate_targets(img)
    finger = finger_image_to_coordinates(finger)
    touchscreen = touchscreen_image_to_coordinate(touchscreen)
    return (finger, touchscreen)


def analyze_dir(path, fps):
    """Process and then analyze a directory of png images.
    
    :param path: The path to the direcotry of images.
    :param fps: The frames per second.
    :return: The latency (seconds) between finger and touchscreen.
    """
    files = [os.path.join(path, o) for o in os.listdir(path) if os.path.isfile(os.path.join(path, o)) and o.endswith('.png')]
    files = sorted(files)
    data = []
    for f in files:
        img = cv2.imread(f)
        data.append(process(img))
    print(data)
    return analyze(data, fps, doPlot=True)


def analyze(data, fps, doPlot=False):
    """Analyze a list of results from :func:`process`.
    
    :param data: The list of results for :func:`process`.
    :param fps: The frames per second.
    :return: The latency (seconds) between finger and touchscreen.
    :raise ValueError: If the data does not contain sufficient information for
        analysis.
    """
    period = 1.0 / float(fps)
    finger_data = []
    touchscreen_data = []
    for idx, (finger, touchscreen) in enumerate(data):
        timestamp = idx * period
        if finger is not None:
            finger_data.append([timestamp, finger[0], finger[1]])
        if touchscreen is not None:
            touchscreen_data.append([timestamp, touchscreen[0], touchscreen[1]])
    finger_data = np.array(finger_data)
    touchscreen_data = np.array(touchscreen_data)
    
    MINIMUM_FRAME_COUNT = 10
    if len(finger_data) < MINIMUM_FRAME_COUNT:
        raise ValueError('finger_data too short')
    if len(touchscreen_data) < MINIMUM_FRAME_COUNT:
        raise ValueError('touchscreen_data too short')

    TIMESTAMP = 0
    X = 1
    Y = 2
    t = finger_data[:, TIMESTAMP]
     
    x0 = [0.0, 0.0, 0.0] # latency in seconds,  offset_x, offset_y
    def func(x, details=False):
        latency, offset_x, offset_y = x
        t2 = touchscreen_data[:, TIMESTAMP] - latency
        touch_x = np.interp(t, t2, touchscreen_data[:, X]) - offset_x
        touch_y = np.interp(t, t2, touchscreen_data[:, Y]) - offset_y
        dx = finger_data[:, X] - touch_x
        dy = finger_data[:, Y] - touch_y
        # Discard data that does not overlap
        invalidDataRange = np.logical_or(t < t2[0], t > t2[-1])
        dx[invalidDataRange] = 0.
        dy[invalidDataRange] = 0.
        residuals = np.hstack((dx, dy))
        if details:
            return {'residuals': residuals, 'touch_x': touch_x, 'touch_y': touch_y}
        return residuals
    rv = scipy.optimize.leastsq(func, x0)
    x = rv[0]
    if doPlot:
        details = func(x, True)
        import matplotlib.pyplot as plt
        plt.plot(t, finger_data[:, X])
        plt.plot(t, details['touch_x'])
        plt.show()
    return x[0]
