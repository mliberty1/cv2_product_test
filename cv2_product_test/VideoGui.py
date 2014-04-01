# Copyright 2014 Jetperch LLC - See LICENSE file.
"""
Display a simple Video GUI using the OpenCV HighGui API.
"""

import cv2
import time
import cv2_product_test.VideoSource
import numpy as np


def onDraw(video_gui, image):
    """Default onDraw callback for :class:`VideoGui`.
    
    :param video_gui: The :class:`VideoGui` instance.
    :param image: The opencv image for possible modification.
    :return: The image to be displayed.  image may be modified in place.
    """
    return image


def onKeyPress(video_gui, ch):
    """Default onKeyPress callback for :class:`VideoGui`.
    
    :param video_gui: The :class:`VideoGui` instance.
    :param ch: The integer character code for the key pressed by the user.
    :return: True if handled, false if unhandled.
    """
    if ch in [27, ord('x'), ord('q')]:
        video_gui.exit = True
    elif ch in [ord(' '), ord('p')]:
        video_gui.paused = not video_gui.paused
    elif ch in [ord('c')]:
        filename = time.strftime('%Y%m%d_%H%M%S', time.gmtime()) + '.png'
        cv2.imwrite(filename, video_gui.frame)
    elif ch in [ord('v')]:
        video_gui.toggle_video_writer()
    elif ch in [ord('b')]:
        video_gui.toggle_image_writer()
    else:
        return False
    return True


class VideoGui(object):
    """Simple class that starts a graphical user interface using the opencv
    HighGUI API to display a video.  The GUI also supports capturing video and
    images using keypresses."""
    
    def __init__(self, video_src, window_name=None):
        """Create a new instance.
        
        :param video_src: The source for the video capture.
        :param window_name: The name for the GUI window.
        """
        self._cap = cv2_product_test.VideoSource.VideoSource(video_src)
        self._myadapter = cv2_product_test.VideoSource.VideoRateAdapter()
        self._cap.add_subscriber(self._myadapter.write)
        #: True to pause the video playback and hold the last frame.
        self.paused = False
        #: False to run, True to exit
        self.exit = True
        #: The current video frame image.
        self.frame = None
        if window_name is None:
            window_name = 'VideoGui'
        self.window_name = window_name
        cv2.namedWindow(self.window_name)
        #: Callable called with (self, image) for each image.
        self.onDraw = None
        #: Callable called with (self, key) for each key pressed.
        self.onKeyPress = onKeyPress
        #: Used to record images to a video file
        self._video_writer = None
        self._frame_size = None
        #: Used to record images to discrete files
        self._image_writer = None

    def get_rectangle(self, rectangle):
        img = self.frame[rectangle.y0:(rectangle.y1 + 1),
                         rectangle.x0:(rectangle.x1 + 1)]
        return img

    @property
    def processed_frames_per_second(self):
        return self._myadapter.fps_out.rate

    def toggle_video_writer(self):
        """Save each frame to a single video file."""
        if self._video_writer is None: # start video recording
            fourcc = 'I420' # 'IYUV' #'YUY2'
            fourcc = cv2.cv.CV_FOURCC(*fourcc)
            print(fourcc)
            fps = int(np.round(self.processed_frames_per_second))
            filename = time.strftime('%Y%m%d_%H%M%S', time.gmtime()) + '.avi'
            self._video_writer = cv2.VideoWriter(filename, fourcc, fps, self._frame_size)
        else: 
            self._video_writer = None

    def toggle_image_writer(self):
        """Save each frame to a unique image file."""
        if self._image_writer is None:
            prefix = time.strftime('%Y%m%d_%H%M%S', time.gmtime())
            self._image_writer = [prefix, 0]
        else:
            self._image_writer = None

    def run(self):
        """Run the VideoGUI.  Blocks until the GUI closes."""
        self.exit = False
        self._cap.start()
        fontFace = cv2.FONT_HERSHEY_PLAIN
        fontScale = 1.0
        fontThickness = 1
        fontSize, _ = cv2.getTextSize('H', fontFace, fontScale, fontThickness)
        while not self.exit:
            frame = self._myadapter.get_most_recent_frame()
            if self.paused:
                frame = self.frame
            self.frame = frame.copy()
            self._frame_size = tuple(frame.shape[1::-1])
            vis = frame.copy()
            if self.onDraw:
                vis = self.onDraw(self, vis)
            fps = 'src_fps=%.2f, proc_fps=%.2f' % (self._cap.fps_source.rate, self._myadapter.fps_out.rate)
            cv2.putText(vis, fps, (10, 10 + fontSize[1]), fontFace, fontScale, (0, 0, 255), fontThickness)
            cv2.imshow(self.window_name, vis)
            if self._video_writer is not None:
                self._video_writer.write(vis)
            if self._image_writer is not None:
                prefix, idx = self._image_writer
                fname = '%s_%06d.png' % (prefix, idx)
                self._image_writer[1] = idx + 1
                cv2.imwrite(fname, vis)
            ch = cv2.waitKey(1)
            if ch == -1:
                continue
            if self.onKeyPress:
                ch &= 0xff 
                if not self.onKeyPress(self, ch):
                    print('Unhandled keypress: %s = %d = 0x%x' % (chr(ch), ch, ch))
        self._cap.join()
        self._myadapter.clear()
        cv2.destroyWindow(self.window_name)
