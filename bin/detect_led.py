#!/usr/bin/env python
# Copyright 2014 Jetperch LLC - See LICENSE file.
"""Detect when an LED is illuminated on product."""

description="""\
Detect when an LED is illuminated on a product.

To use this script, first press the left mouse button and drag the rectangle
over the product.  Release the mouse button once the product is selected.
Then press the right mouse button and drag the rectangle over just the LED.
Release the mouse button once the LED is selected.  The script will then
track the product, the LED location and indicate when the LED is illuminated.

Press 'x' or 'ESC' to exit.
Press space to pause and resume the video which makes drawing rectangles easier.
"""

epilog = """\
Copyright 2014 Jetperch LLC
"""

"""
Based upon the following files:
https://github.com/Itseez/opencv/blob/master/samples/python2/common.py
https://github.com/Itseez/opencv/blob/master/samples/python2/feature_homography.py
https://github.com/Itseez/opencv/blob/master/samples/python2/plane_tracker.py 
"""

import cv2
import numpy as np
import os
import sys
import time
import argparse


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from cv2_product_test.SelectRectangularRegion import SelectRectangularRegion
from cv2_product_test.find import Features
from cv2_product_test.VideoGui import VideoGui, onKeyPress


def get_parser():
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=epilog)
    parser.add_argument('video_src',
                        nargs='?',
                        default=0,
                        type=int,
                        help='The integer video source number.')
    return parser

class DetectLed(object):
    """Detect if the region containing an LED in the specified product is
    illuminated.  Output the result to the video feed and standard output."""
    
    def __init__(self, window_name, features_search_spec, threshold):
        """Initialize a new instance.
        
        :param window_name: The cv2.namedWindow name.
        :param features_search_spec:
        :param threshold:
        """
        self._window_name = window_name
        self._select = SelectRectangularRegion(self.rectangular_region_callback)
        self._features = Features(features_search_spec)
        self._target = None
        self._led = None
        self.threshold = threshold
        self._isLedOn = False
        cv2.setMouseCallback(self._window_name, self._select.onMouse)
        self.frame = None 
    
    def rectangular_region_callback(self, button_name, rectangle):
        """Callback for :class:`SelectRectangularRegion` which allows the user
        to select the product region using left mouse drag and then the LED
        region using right mouse drag."""
        print('rectangular_region_callback(%s, %s)' % (button_name, rectangle))
        if button_name == 'left':
            image = rectangle.extract(self.frame)
            self._features.clear()
            self._target = None
            self._led = None
            try:
                self._features.add_target(image, {})
            except ValueError:
                print('Region not added')
        elif button_name == 'right':
            if not self._target:
                print('No product defined: use left mouse button.')
            self._led = self._target.inTargetCoordinates(rectangle.quad)

    def ledValue(self, image, quad):
        """Robustly compute the value of the LED region.
        
        :param image: The image to search.
        :param quad: The quadrilateral points for the LED region inside image.
        :return: The LED metric used to determine on/off.
        """
        quad = np.float32([quad])
        x, y, width, height = cv2.boundingRect(quad)
        img = image[y:(y+height), x:(x+width)]
        if img is None or np.prod(img.shape) == 0:
            return 0
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        quad = quad.reshape(-1, 2)
        quad -= np.float32([x, y])
        mask = np.zeros(img.shape, np.uint8)
        quad = np.int32(quad.reshape(1, -1, 2))
        cv2.drawContours(mask, quad, 0, (1), cv2.cv.CV_FILLED)
        # Use percentile matching for statistical robustness
        values = img[mask != 0]
        if len(values) < 5:
            v2 = 0
        else:
            v2 = np.percentile(values, 90)
        return v2

    def onDraw(self, video_gui, image):
        """Callback from :class:`VideoGui` to draw our results on the image.
        
        :param video_gui: The :class:`VideoGui` instance.
        :param image: The image for drawing which can be modified in place.
        :return: image.
        """
        self.frame = image
        image = self._select.onDraw(image)
        tracked = self._features.find(image)
        if tracked is not None and len(tracked):
            self._target = tracked[0]
        if self._target is not None:
            cv2.polylines(image, [np.int32(self._target.quad)], True, (255, 255, 255), 2)
            if self._led is not None:
                quad = self._target.inImageCoordinates(self._led)
                cv2.polylines(image, [np.int32(quad)], True, (0, 255, 0), 2)
                v = self.ledValue(image, quad)
                ymax = image.shape[0]
                v_str = '%3d' % v
                sz, _ = cv2.getTextSize(v_str, cv2.FONT_HERSHEY_PLAIN, 1.0, 1)
                cv2.putText(image, v_str, (10, ymax - 10), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 1)
                if v >= self.threshold:
                    cv2.putText(image, 'ON', (10, ymax - 10 - sz[1] - 10), cv2.FONT_HERSHEY_PLAIN, 3.0, (255, 255, 255), 4)
                    if not self._isLedOn:
                        print('%.3f: LED ON' % time.time())
                        self._isLedOn = True
                elif self._isLedOn:
                    print('%.3f: LED OFF' % time.time())
                    self._isLedOn = False
        return image
    
    
    def onKeyPress(self, video_gui, ch):
        return onKeyPress(video_gui, ch)


if __name__ == '__main__':
    args = get_parser().parse_args()
    window_name = 'Detect LED'
    gui = VideoGui(args.video_src, window_name)
    detectLed = DetectLed(window_name, features_search_spec='orb-flann', threshold=215)
    gui.onDraw = detectLed.onDraw
    gui.onKeyPress = detectLed.onKeyPress
    gui.run()
