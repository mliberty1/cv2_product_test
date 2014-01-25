# Copyright 2014 Jetperch LLC - See LICENSE file.
"""
Find images using a variety of techniques.

This software is heavily based upon and copies some code from the following
opencv samples:

* https://github.com/Itseez/opencv/blob/master/samples/python2/find_obj.py
* https://github.com/Itseez/opencv/blob/master/samples/python2/feature_homography.py
* https://github.com/Itseez/opencv/blob/master/samples/python2/plane_tracker.py 
"""

import cv2
import numpy as np
from collections import namedtuple


class FindError(Exception):
    """Error indicating that find failed."""
    def __init__(self, metric, location, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)
        self.metric = metric
        self.location = location


FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
FLANN_INDEX_LSH    = 6
FLANN_PARAMS_NORM  = {
                      'algorithm': FLANN_INDEX_KDTREE, 
                      'trees': 5
                     }
FLANN_PARAMS_HAMMING = {
                        'algorithm':    FLANN_INDEX_LSH,
                        'table_number': 6,
                        'key_size':     12,
                        'multi_probe_level': 1
                       }
MIN_MATCH_COUNT_DEFAULT = 10


Target = namedtuple('Target', 'image, quad, keypoints, descriptors, data')
"""The Target for image searches.  
  image     - image to track
  quad      - Target boundary quad in the original image
  keypoints - keypoints detected inside rect
  descrs    - their descriptors
  data      - some user-provided data
"""


class TrackedTarget(object):
    """A Target that was found during an image search.

    :var target: reference to :class:`Target`
    :var image: The image used for the search.
    :var inliers: The list of matching points given as [((x0, y0), (x1, y1)), ...]
    :var outliers: The list of unmatched points.
    :var H: homography matrix from (x0, y0) to (x1, y1)
    :var quad: target bounary quad in input frame
    """
    def __init__(self, target, image, inliers, outliers, H, quad):
        self.target = target
        self.image = image
        self.inliers = inliers
        self.outliers = outliers
        self.H = H
        self.quad = quad

    def inTargetCoordinates(self, p):
        p = np.float32(p)
        H = np.linalg.inv(self.H)
        p = cv2.perspectiveTransform(p.reshape(1, -1, 2), H).reshape(-1, 2)
        return p
    
    def inImageCoordinates(self, p):
        p = np.float32(p)
        p = cv2.perspectiveTransform(p.reshape(1, -1, 2), self.H).reshape(-1, 2)
        return p        

    def draw(self, image, color, width, quad=None):
        """Draw the tracked target on the image.
        
        :param image: The image used to draw the tracked target.  If None,
            then use the same image from the search.
        :param color: The color tuple.
        :param width: The line width in pixels.
        :param quad: The quad to draw.  When None (default), uses the quad from
            this instance.
        :returns: The image, which is also modified in place. 
        """
        if image is None:
            image = self.image.copy()
        if quad is None:
            quad = self.quad
        color = tuple(np.int32(color).tolist())
        cv2.polylines(image, [np.int32(quad)], True, color, int(width))
        return image

    def visualize(self):
        """Visualize the search results.
    
        :returns: The resulting visualization image.
        """
        colors = {'outline': (220, 220, 220),
                  'inlier': (0, 255, 0),
                  'outlier': (0, 0, 255),
                  'lines': (128, 220, 128)}
        # Create output image for visualization
        gap = 5
        h1, w1 = self.target.image.shape[:2]
        h2, w2 = self.image.shape[:2]
        vis = np.zeros((max(h1, h2), w1 + w2 + gap, 3), np.uint8)
        vis[:h1, :w1, :] = self.target.image
        w1 += gap
        vis[:h2, w1:w1+w2, :] = self.image
    
        # Draw the located object    
        quad = np.float32(self.quad) + np.float32([w1, 0])
        self.draw(vis, colors['outline'], 2, quad)
    
        # draw point details
        inliers  = [(x0, y0, x1 + w1, y1) for (x0, y0), (x1, y1) in self.inliers]
        outliers = [(x0, y0, x1 + w1, y1) for (x0, y0), (x1, y1) in self.outliers]
        if colors['outlier'] is not None: # draw x on each point
            r = 2 # radius
            thickness = 2
            for x0, y0, x1, y1 in outliers:
                cv2.line(vis, (x0 - r, y0 - r), (x0 + r, y0 + r), colors['outlier'], thickness)
                cv2.line(vis, (x0 + r, y0 - r), (x0 - r, y0 + r), colors['outlier'], thickness)
                cv2.line(vis, (x1 - r, y1 - r), (x1 + r, y1 + r), colors['outlier'], thickness)
                cv2.line(vis, (x1 + r, y1 - r), (x1 - r, y1 + r), colors['outlier'], thickness)
        if colors['lines'] is not None:
            for x0, y0, x1, y1 in inliers:
                cv2.line(vis, (x0, y0), (x1, y1), colors['lines'], 1)
        if colors['inlier'] is not None:
            for x0, y0, x1, y1 in inliers:
                cv2.circle(vis, (x0, y0), 2, colors['inlier'], -1)
                cv2.circle(vis, (x1, y1), 2, colors['inlier'], -1)
        return vis    


class Features(object):
    """Find an image using feature-based object recognition.
    
    This class holds initialized search objects to accelerate subsequent 
    searchs which is especially useful for processing video streams.
    """
    
    def __init__(self, search_spec=None):
        """Initialize the feature detection framework.
        
        :param search_spec: The specification string which consists of 
            "[detector]-[matcher]".  Valid detector values include
            sift, surf (Default), and orb.  Valid matcher values include flann and
            bf (brute_force) (Default).
        """
        if search_spec is None or not search_spec:
            search_spec = 'orb'
        chunks = search_spec.split('-')
        if len(chunks) == 1:
            chunks.append('bf')
        elif len(chunks) != 2:
            raise ValueError('Invalid search specification: %s' % search_spec)
        detector_name, matcher_name = [x.lower() for x in chunks]
        self._detector, norm = self._init_detector(detector_name)
        self._matcher = self._init_matcher(matcher_name, norm)
        self._targets = []
        self.min_match_count = MIN_MATCH_COUNT_DEFAULT
    
    def _init_detector(self, name):
        norm = cv2.NORM_L2
        if name == 'fast':
            detector = cv2.FastFeatureDetector()
        elif name == 'brisk':
            detector = cv2.BRISK()
        elif name == 'sift':
            detector = cv2.SIFT()
        elif name == 'surf':
            detector = cv2.SURF(800)
        elif name == 'orb':
            detector = cv2.ORB(1200)
            norm = cv2.NORM_HAMMING
        else:
            raise ValueError('Unsupported detector: %s' % name)
        return detector, norm

    def _init_matcher(self, name, norm):
        if name == 'flann':
            if norm == cv2.NORM_L2:
                flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            else:
                flann_params = dict(algorithm = FLANN_INDEX_LSH,
                                    table_number = 6, # 12
                                    key_size = 12,     # 20
                                    multi_probe_level = 1) #2
            matcher = cv2.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
        elif name in ['bf', 'brute_force']:
            matcher = cv2.BFMatcher(norm)
        else:
            raise ValueError('Unsupported matcher: %s' % name)
        return matcher

    def add_target(self, image, data=None):
        """Add a new target to the serach list.
        
        :param image: The image containing just the target.
        :param data: Additional user data.  Defaults to None.
        :raises ValueError: If the image is insufficient.
        """
        keypoints, descriptors = self._detector.detectAndCompute(image, None)
        if len(keypoints) < self.min_match_count:
            raise ValueError('Target image has insufficient keypoints')
        y, x = image.shape[:2]
        quad = np.float32([[0, 0], [x, 0], [x, y], [0, y]])
        self._targets.append(Target(image, quad, keypoints, descriptors, data))
        self._matcher.add([descriptors])
        #print('Target has %d features' % len(keypoints))

    def clear(self):
        """Remove all targets"""
        self._targets = []
        self._matcher.clear()

    def draw_keypoints(self, image, color=None):
        if color is None:
            color = [0, 255, 0]
        keypoints, _ = self._detector.detectAndCompute(image, None)
        image = image.copy()
        return cv2.drawKeypoints(image, keypoints, color=color)

    def find(self, image, k=None, ratio=None):
        """Find the targets in the provided image.
        
        :param image: The image to search for targets.
        :param k: The number of knnMatches to use.  None (Default) uses 2.
        :param ratio: The distance ration to use.  None (Default) uses 0.75.
        :returns: A list containing :class:`TrackedTarget` instances for
            each target found.
        """
        if not self._targets:
            return []
        k = 2 if k is None else k
        ratio = 0.75 if ratio is None else ratio
        keypoints, descriptors = self._detector.detectAndCompute(image, None)
        if len(keypoints) < self.min_match_count:
            return []
        matches = self._matcher.knnMatch(descriptors, k=int(k))
        matches = [m[0] for m in matches if len(m) == 2 and m[0].distance < m[1].distance * ratio]
        if len(matches) < self.min_match_count:
            return []
        matches_by_id = [[] for _ in xrange(len(self._targets))]
        for m in matches:
            matches_by_id[m.imgIdx].append(m)
        tracked = []
        for imgIdx, matches in enumerate(matches_by_id):
            if len(matches) < self.min_match_count:
                continue
            target = self._targets[imgIdx]
            p0 = [target.keypoints[m.trainIdx].pt for m in matches]
            p1 = [keypoints[m.queryIdx].pt for m in matches]
            p0, p1 = np.float32((p0, p1))
            H, status = cv2.findHomography(p0, p1, cv2.RANSAC, 3.0)
            status = status.ravel() != 0
            if status.sum() < self.min_match_count:
                continue
            p0, p1 = np.int32((p0, p1))
            inliers  = [((x0, y0), (x1, y1)) for (x0, y0), (x1, y1), s in zip(p0, p1, status) if s]
            outliers = [((x0, y0), (x1, y1)) for (x0, y0), (x1, y1), s in zip(p0, p1, status) if not s]
            quad = cv2.perspectiveTransform(target.quad.reshape(1, -1, 2), H).reshape(-1, 2)
            track = TrackedTarget(target=target, image=image, inliers=inliers, outliers=outliers, H=H, quad=quad)
            tracked.append(track)
        tracked.sort(key = lambda t: len(t.inliers), reverse=True)
        return tracked
        
    def onDraw(self, image, tracked=None):
        if tracked is None:
            tracked = self.search(image)
        for tr in tracked:
            tr.draw(image, (255, 255, 255), 2)
        return image


def find(image_to_find, image, search_spec=None, **kwargs):
    """Search an image to see if it contains the provided reference image.

    Additional arguments are passed to the search.
    
    :param image_to_find: The reference image to find in image.
    :param image: The image to search.
    :param search_spec: The search specification.  A value of 'template' invokes
        :func:`search_template`.  All other values are defined in 
        :class:`Features`.
    :returns: A :class:`TrackedTarget` instance if found or None if not found.
    """
    if search_spec is not None and search_spec.startswith('template'):
        return _find_using_template(image_to_find, image, **kwargs)
    features = Features(search_spec)
    features.add_target(image_to_find)
    tracked = features.find(image, kwargs.get('k'), kwargs.get('ratio'))
    if not len(tracked):
        raise FindError(1.0, None)
    return tracked[0]


def _find_using_template(image_to_find, image, threshold=None, **kwargs):
    """Search an image to see if it contains the provided reference image.
    
    :param image_to_find: The reference image to find in image.
    :param image: The image to search.
    :param threshold: The metric threshold.  1e-6 (Default) finds nearly 
        exact images undistorted by noise or error.
    :returns: A :class:`TrackedTarget` instance if found.
    :raises FindError: If the image_to_find could not be found in image.
    """
    threshold = 1e-6 if threshold is None else threshold
    result = cv2.matchTemplate(image, image_to_find, cv2.TM_SQDIFF_NORMED)
    idx = np.argmin(result)
    metric = np.ravel(result)[idx]
    x0, y0 = np.unravel_index(idx, result.shape)[-1::-1]
    if metric > threshold:
        raise FindError(metric, (x0, y0))
    x, y = image_to_find.shape[1::-1]
    target = Target(image_to_find, [[0, 0], [x, 0], [x, y], [0, y]], None, None, None)
    x1 = x0 + image_to_find.shape[1]
    y1 = y0 + image_to_find.shape[0]
    quad = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    H = np.array([[0., 0., x0], [0., 0., y0], [0., 0., 1.0]])
    return TrackedTarget(target, image, [(0, 0)], [(x0, y0)], H, quad)
