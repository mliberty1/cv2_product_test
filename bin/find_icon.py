#!/usr/bin/env python
# Copyright 2014 Jetperch LLC - See LICENSE file.
"""Find an icon in an image."""

description="""\
Find an icon in an image.

This script will search an image attempting to match an icon.  The script 
supports a number of different methods.  Most methods are 2-D feature based,
but template-based matching is also supported.  See the OpenCV documentation
for details on each method:

http://docs.opencv.org/modules/features2d/doc/features2d.html
http://docs.opencv.org/modules/imgproc/doc/object_detection.html
"""

epilog = """\
An example for using this command is
    bin/find_icon.py cv2_product_test/test/android.png cv2_product_test/test/android_camcorder.png sift-flann

Copyright 2014 Jetperch LLC
"""

import sys
import os
import argparse
import cv2

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
import cv2_product_test.find


def get_parser():
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=epilog)
    parser.add_argument('image',
                        help='The image filename to search for the icon.')
    parser.add_argument('icon',
                        help='The icon filename to find in image.')
    parser.add_argument('spec',
                        help='The search specification which includes '
                             'sift-flann, surf-flann, orb-flann, template.')
    return parser

if __name__ == '__main__':
    args = get_parser().parse_args()
    icon = cv2.imread(args.icon)
    image = cv2.imread(args.image)
    rv = 0
    try:
        r = cv2_product_test.find.find(icon, image, args.spec)
        print('SUCCESS: icon found')
        sys.exit(0)
    except cv2_product_test.find.FindError:
        print('ERROR: icon not found')
        sys.exit(1)
