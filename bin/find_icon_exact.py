#!/usr/bin/env python
# Copyright 2014 Jetperch LLC - See LICENSE file.
"""Find an icon in an image.
Usage: find_icon_exact [image] [icon]

::

    bin/find_icon_exact.py cv2_product_test/test/android.png cv2_product_test/test/android_camcorder.png

"""

import cv2
import sys
import numpy as np


def saveMatchResult(filename, result):
    result = result * (255.0 / np.max(result)) # scale image to 0.0 to 1.0
    result = result.astype(np.uint8)
    cv2.imwrite(filename, result)
    

if __name__ == '__main__':
    # Parse arguments
    try:
        _, icon_filename, image_filename = sys.argv
    except:
        print(__doc__)
        sys.exit(2)
    threshold = 1e-6
    icon = cv2.imread(icon_filename)
    image = cv2.imread(image_filename)
    
    # Match the icon template against the image
    result = cv2.matchTemplate(image, icon, cv2.TM_SQDIFF_NORMED)
    #saveMatchResult('match.png', result) 
    idx = np.argmin(result)
    metric = np.ravel(result)[idx]
    if metric > threshold:
        print('ERROR: %s' % metric)
        sys.exit(1)
    coord = np.unravel_index(idx, result.shape)[-1::-1]
    print('SUCCESS: %s at %s' % (metric, coord))
    sys.exit(0)
