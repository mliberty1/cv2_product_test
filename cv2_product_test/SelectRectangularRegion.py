# Copyright 2014 Jetperch LLC - See LICENSE file.
"""
Support region selection on drag operations for each mouse button.
"""

import cv2


class Rectangle(object):
    """Class which holds a single rectangle defined by opposite points.
    
    Note that the points are inclusive.
    """
    
    def __init__(self, x0, y0, x1, y1):
        #: The x-axis coordinate of the first point (usually upper left).
        self.x0 = x0
        #: The y-axis coordinate of the first point (usually upper left). 
        self.y0 = y0
        #: The x-axis coordinate of the second point (usually bottom right). 
        self.x1 = x1
        #: The y-axis coordinate of the second point (usually bottom right). 
        self.y1 = y1

    @property
    def quad(self):
        """Get the list of [x, y] vertices defining the quadrilatoral."""
        return [[self.x0, self.y0], [self.x1, self.y0], [self.x1, self.y1], [self.x0, self.y1]]

    @property
    def upper_left(self):
        return (min(self.x0, self.x1), min(self.y0, self.y1))

    @property
    def bottom_right(self):
        return (max(self.x0, self.x1), max(self.y0, self.y1))

    @property
    def width(self):
        return self.x1 - self.x0 + 1

    @property
    def height(self):
        return self.y1 - self.y0 + 1

    def extract(self, image):
        """Extract a rectangular region from an image.
        
        :param self: The :class:`Rectangle` instance defining the region.
        :param image: The source image.
        :return: The extract image.
        """ 
        return image[self.y0:(self.y1 + 1),
                     self.x0:(self.x1 + 1)]
        
    def replace(self, image, replacement_image):
        width = min(self.width, replacement_image.shape[1])
        height = min(self.height, replacement_image.shape[0])
        image[self.y0:(self.y0 + height), self.x0:(self.x0 + width), :] = \
            replacement_image[:height, :width, :]


#               button down event,     button up event,     name,     color
_BUTTON_MAP = [(cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONUP, 'left',   [255, 255,   0]),
               (cv2.EVENT_RBUTTONDOWN, cv2.EVENT_RBUTTONUP, 'right',  [0,   255,   0]),
               (cv2.EVENT_MBUTTONDOWN, cv2.EVENT_MBUTTONUP, 'middle', [0,   255, 255])]
"""The mapping from opencv events to button names."""


class SelectRectangularRegion(object):
    """Allow the user to select a rectangular region by pushing a mouse button
    dragging the pointer, and releasing the mouse button."""
    
    
    def __init__(self, callback):
        """Create a new instance.
        
        To create and activate the selection on a window named 'mywindow'
        
        ::
            
            select = SelectRectangularRegion(mycallback)
            cv2.setMouseCallback('mywindow', select.onMouse) 
        
        :param callback: The callback(button_name, :class:`Rectangle`) that is
            called when the user creates a region.
        """ 
        self._callback = callback
        self._start = [None, None, None]
        self._current = None

    def onMouse(self, event, x, y, flags, param):
        """Callback for cv2.setMouseCallback"""
        self._current = (x, y)
        for idx, (down_event, up_event, name, _) in enumerate(_BUTTON_MAP):
            if event == down_event:
                self._start[idx] = self._current
            if event == up_event:
                rectangle = Rectangle(self._start[idx][0], self._start[idx][1], 
                                      self._current[0], self._current[1])
                self._callback(name, rectangle)
                self._start[idx] = None
        
    def onDraw(self, image):
        """Draw any current rectangles on the provided image."""
        for idx, start in enumerate(self._start):
            if start:
                color = _BUTTON_MAP[idx][3]
                cv2.rectangle(image, start, self._current, color, 2)
        return image
