#!/usr/bin/env python
# Copyright 2014 Jetperch LLC - See LICENSE file.
"""
Measure the touchscreen latency.  

This application is intended for use with a very simple Android application
which displays a white background with a magenta dot at the current touch 
location.  To measure latency, aim a video camera at the Android device.
Using only your pointer finger, touch the screen ensuring that only your
pointer finger overlaps the display.  
"""

import cv2
import os
import sys
import time


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from cv2_product_test.SelectRectangularRegion import SelectRectangularRegion
from cv2_product_test.VideoGui import VideoGui, onKeyPress
import cv2_product_test.latency 


class MeasureTouchscreenLatency(object):
    
    def __init__(self, window_name, features_search_spec, threshold):
        """Initialize a new instance.
        
        :param window_name: The cv2.namedWindow name.
        """
        self._window_name = window_name
        self._select = SelectRectangularRegion(self.rectangular_region_callback)
        cv2.setMouseCallback(self._window_name, self._select.onMouse)
        self._rectangle = None
        self._data = []
        self._last_active = time.time()
        #: The duration (seconds) before latency data colleciton resets
        self.inactivity_timer = 1.0
        #: The current latency estimate (seconds)
        self._latency = None
    
    def rectangular_region_callback(self, button_name, rectangle):
        """Callback for :class:`SelectRectangularRegion` which allows the user
        to select the product region using left mouse drag and then the LED
        region using right mouse drag."""
        print('rectangular_region_callback(%s, %s)' % (button_name, rectangle))
        if button_name == 'left':
            self._rectangle = rectangle

    def reset(self):
        self._data = []
        self._last_active
        self._latency = None

    def onDraw(self, video_gui, image):
        """Callback from :class:`VideoGui` to draw our results on the image.
        
        :param video_gui: The :class:`VideoGui` instance.
        :param image: The image for drawing which can be modified in place.
        :return: image.
        """
        self._select.onDraw(image)
        if self._rectangle is None:
            return image
        # Extract the region and process
        img = self._rectangle.extract(image)
        d = cv2_product_test.latency.process(img)
        self._data.append(d)
        
        # Draw the identified targets
        cv2.rectangle(image, self._rectangle.upper_left, self._rectangle.bottom_right, (255, 128, 128), 2)
        active = False
        for location, color in zip(d, [(0, 255, 0), (0, 0, 255)]):
            if location is not None:
                p = (self._rectangle.x0 + location[0], self._rectangle.y0 + location[1])
                cv2.circle(image, p, 4, color, -1)
                active = True
        if active: # analyze the data
            try:
                self._latency =  cv2_product_test.latency.analyze(self._data, video_gui.processed_frames_per_second)
            except ValueError:
                pass # Not enough data yet.
            except Exception as ex:
                print(ex) # attempt to keep going.
            self._last_active = time.time()
        elif time.time() - self._last_active > self.inactivity_timer:
            self.reset()
        if self._latency: # display the latency
            s = 'Latency is %d ms' % int(self._latency * 1000)
            y = image.shape[0]
            cv2.putText(image, s, (10, y - 10), cv2.FONT_HERSHEY_PLAIN, 3.0, (0, 0, 255), 2)
        return image
    
    def onKeyPress(self, video_gui, ch):
        return onKeyPress(video_gui, ch)


if __name__ == '__main__':
    print __doc__
    try: 
        video_src = sys.argv[1]
    except: 
        video_src = 0
    window_name = 'Measure touchscreen latency'
    gui = VideoGui(video_src, window_name)
    latency = MeasureTouchscreenLatency(window_name, features_search_spec='orb-flann', threshold=215)
    gui.onDraw = latency.onDraw
    gui.onKeyPress = latency.onKeyPress
    gui.run()
