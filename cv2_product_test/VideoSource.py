# Copyright 2014 Jetperch LLC - See LICENSE file.
"""
Gracefully handle video input and allow processing blocks to run
slower than the full video frame rate.
"""

import cv2
import threading
import Queue
import time
import logging
log = logging.getLogger(__name__)


class EventsPerSecond(object):
    """Compute the number rate of events per second.
    
    :var rate: The computed rate in events per second.
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset the instance."""
        self._start = time.time()
        self._count = 0
        self.rate = 0
    
    def event(self):
        """Signal that a new event has occurred and update stats."""
        t = time.time()
        self._count += 1
        dt = t - self._start
        if dt > 1.0:
            self.rate = self._count / dt
            self._start = t
            self._count = 0


class VideoRateAdapter(object):
    """Connect a video source to a video processor while allowing the processor
    to gracefully skip frames."""
    
    def __init__(self):
        self.drops = 0
        self.fps_in = EventsPerSecond()
        self.fps_out = EventsPerSecond()
        self._queue = Queue.Queue()
    
    def write(self, frame):
        self.fps_in.event()
        self._queue.put(frame)
    
    def clear(self):
        """Empty the queue."""
        try:
            while True:
                self._queue.get(False)        
        except Queue.Empty:
            pass
    
    def get(self, timeout=2.0):
        return self._queue.get(timeout=timeout)

    def get_most_recent_frame(self, timeout=2.0):
        frame = None
        drops = 0
        try:
            while True:
                frame = self._queue.get(False)
                drops += 1
        except Queue.Empty:
            pass
        if frame is None:
            frame = self._queue.get(timeout=timeout)
        else:
            drops -= 1
        self.fps_out.event()
        self.drops += drops
        return frame


class VideoSource(threading.Thread):
    """A video source that grabs every frame and dispatches each frame to the
    connected subscriber."""
    
    def __init__(self, video_src):
        """Create a new instance.
        
        :param video_src: The source for the video capture.
        """
        threading.Thread.__init__(self)
        self._cap = cv2.VideoCapture(video_src)
        self._quit = False
        self.fps_source = EventsPerSecond()
        self._subscribers = []

    def add_subscriber(self, subscriber):
        """Add a subscriber for this video stream.
        
        :param subscriber: The subscriber to add.  A subscriber is a callable
            which accepts a single argument, the next frame image.
        """
        self._subscribers.append(subscriber)
        
    def remove_subscriber(self, subscriber):
        """Remove a subscriber from this video stream.
        
        :param subscriber: The subscriber to remove.
        """
        self._subscribers.remove(subscriber)

    def start(self):
        """Start streaming video from a new thread."""
        self._quit = False
        self.fps_source.reset()
        threading.Thread.start(self)

    def join(self, timeout=None):
        """Rejoin the streaming thread and stop streaming."""
        self._quit = True
        rc = threading.Thread.join(self, timeout)
        return rc
    
    def run(self):
        """Perform the main thread loop."""
        self._cap.set(cv2.cv.CV_CAP_PROP_GAIN, 0.0)
        while not self._quit:
            if not self._cap.grab():
                log.error("cv2.VideoCapture.grab() failed")
                break
            ret, frame = self._cap.retrieve()
            if not ret:
                log.error("cv2.VideoCapture.retrieve() failed")
                break
            self.fps_source.event()
            for subscriber in self._subscribers:
                subscriber(frame)
