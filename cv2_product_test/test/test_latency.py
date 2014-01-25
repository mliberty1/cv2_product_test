# Copyright 2014 Jetperch LLC - See LICENSE file.

import cv2_product_test.latency
import unittest
import os
import cv2
import numpy as np


MY_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(MY_DIR, 'latency')


IMAGES = [
          # img, filename,    finger,   touchscreen
          [None, '000010.png', ( 66, 22), (144,  44)],
          [None, '000046.png', (160, 69), (121, 101)],
          [None, '000050.png', (182, 46), None],
          [None, '000021.png', (170, 94), (127, 85)],
         ]
for entry in IMAGES:
    entry[0] = cv2.imread(os.path.join(DATA_DIR, entry[1]))


class TestLatency(unittest.TestCase):

    def test_empy_image(self):
        x = np.zeros((0, 0, 3), dtype=np.uint8)
        v = cv2_product_test.latency.process(x)
        self.assertEquals(v, (None, None))
    
    def test_blank_image(self):
        x = np.ones((200, 200, 3), dtype=np.uint8) * 255
        v = cv2_product_test.latency.process(x)
        self.assertEquals(v, (None, None))
    
    def test_process_images(self):
        for image, filename, exp_finger, exp_touchscreen in IMAGES:
            finger, touchscreen = cv2_product_test.latency.process(image)
            self.assertEquals(touchscreen, exp_touchscreen)
            self.assertEquals(finger, exp_finger)

    @unittest.SkipTest
    def test_development_experiment(self):
        import scipy.optimize
        latency = 0.1
        t = np.arange(0, 2, 0.05)
        x0 = np.sin(2 * np.pi * t)
        x1 = np.sin(2 * np.pi * (t - latency))
        def func(x, details=False):
            t2 = t - x[0]
            y = np.interp(t, t2, x1)
            r = x0 - y
            if details:
                return {'r': r, 'y': y}
            return r
        rv = scipy.optimize.leastsq(func, [0.0])
        details = func(rv[0], True)
        print(rv[0])
        import matplotlib.pyplot as plt
        plt.plot(t, x0)
        plt.plot(t, details['y'])
        plt.show()

    def make_data(self, latency, fps, offset):
        data = []
        frames = 100.
        duration = frames / float(fps)
        t = np.arange(0, duration, 1. / fps)
        x0p = np.sin(2 / duration * np.pi * t)
        y0p = np.cos(3 / duration * np.pi * t)
        x1p = np.sin(2 / duration * np.pi * (t - latency)) + offset[0]
        y1p = np.cos(3 / duration * np.pi * (t - latency)) + offset[1]
        for x0, y0, x1, y1 in zip(x0p, y0p, x1p, y1p):
            data.append( [(x0, y0), (x1, y1)] )
        return data

    def test_analyze_no_offset(self):
        fps = 30.0
        for latency in [0, 0.1, 0.5, 1.0]:
            data = self.make_data(latency, fps, (0, 0))
            x = cv2_product_test.latency.analyze(data, fps)
            np.testing.assert_allclose(x, latency, rtol=1e-3, atol=1e-3)

    def test_analyze_offset(self):
        fps = 30.0
        latency = 0.1
        data = self.make_data(latency, fps, (10.0, 20.0))
        x = cv2_product_test.latency.analyze(data, fps)
        np.testing.assert_allclose(x, latency, rtol=1e-3, atol=1e-3)

    def test_analyze_missing_data(self):
        fps = 30.0
        latency = 0.1
        data = self.make_data(latency, fps, (10.0, 20.0))
        for touchscreen_missing in [0, 1, 10, 20, 30, -1, -2, -3, -4]:
            data[touchscreen_missing][1] = None
        x = cv2_product_test.latency.analyze(data, fps)
        np.testing.assert_allclose(x, latency, rtol=1e-3, atol=1e-3)
            