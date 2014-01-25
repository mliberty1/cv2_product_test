# Copyright 2014 Jetperch LLC - See LICENSE file.

import cv2_product_test.find
import unittest
import os
import cv2
import numpy as np


DATA_DIR = os.path.dirname(os.path.abspath(__file__))

IMG1 = cv2.imread(os.path.join(DATA_DIR, 'icons.png'))
TGT1 = cv2.imread(os.path.join(DATA_DIR, 'icons_region.png'))
EXP1 = [[172, 62], [270, 62], [270, 163], [172, 163]]

IMG2  = cv2.imread(os.path.join(DATA_DIR, 'android.png'))
TGT2A = cv2.imread(os.path.join(DATA_DIR, 'android_camcorder.png'))
EXP2B = [[1170, 632], [1305, 632], [1305, 788], [1170, 788]]
TGT2B = cv2.imread(os.path.join(DATA_DIR, 'android_firefox.png'))


class TestFeatures(unittest.TestCase):
    
    def test_sift_flann(self):
        s = cv2_product_test.find.find(TGT1, IMG1, 'sift-flann')
        self.assertIsNotNone(s)
        np.testing.assert_allclose(s.quad, EXP1, atol=5.0)
        vis = s.visualize()
        #cv2.imwrite('tmp.png', vis)

    def test_surf_flann(self):
        s = cv2_product_test.find.find(TGT1, IMG1, 'surf-flann')
        self.assertIsNotNone(s)
        np.testing.assert_allclose(s.quad, EXP1, atol=5.0)
    
    def test_surf_flann_2(self):
        s = cv2_product_test.find.find(TGT2B, IMG2, 'sift-flann')
        self.assertIsNotNone(s)
        #np.testing.assert_allclose(s.quad, EXP2B, atol=5.0)
    
    def test_template(self):
        s = cv2_product_test.find.find(TGT2B, IMG2, 'template')
        self.assertIsNotNone(s)
        np.testing.assert_allclose(s.quad, EXP2B)
    
    @unittest.SkipTest
    def test_draw_keypoints(self):
        f = cv2_product_test.find.Features('sift-flann')
        img = f.draw_keypoints(TGT2B)
        cv2.namedWindow('draw_keypoints')
        cv2.imshow('draw_keypoints', img)
        cv2.waitKey(0)
