# Copyright 2014 Jetperch LLC - See LICENSE file.

from cv2_product_test.SelectRectangularRegion import Rectangle
import unittest


class TestRectangle(unittest.TestCase):
    
    def setUp(self):
        self.r1 = Rectangle(1, 2, 11, 17)
    
    def test_constructor(self):
        self.assertEqual(self.r1.x0, 1)
        self.assertEqual(self.r1.y0, 2)
        self.assertEqual(self.r1.x1, 11)
        self.assertEqual(self.r1.y1, 17)
    
    def test_width(self):
        self.assertEqual(self.r1.width, 11)
        self.assertEqual(self.r1.height, 16)
        
    def test_quad(self):
        self.assertEqual(self.r1.quad, [[1, 2], [11, 2], [11, 17], [1, 17]])