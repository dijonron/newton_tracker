import matplotlib.pyplot as plt
import cv2 as cv
from cv2 import VideoCapture
from matplotlib.pyplot import figure, flag
import numpy as np
import time


class NewtonTracker:
    """Newton Tracker

    """

    def __init__(self):
        self.use_mask = False
        self.img = None
        self.mask = None
        self.image_window = "Source Image"
        self.templ_window = "Template Window"
        self.diff_window = "Diff Window"
        self.match_method = 0

        self.cap = None
        self.prev_frame = None
        self.prev_gray = None
        self.roi = None
        self.templ = None

    def setMatchMethod(self, match_method):
        self.match_method = match_method

    def load_video_sequence(self, filename: str):
        """Load and return the (.mp4) video specified by ``filename``.

            @arg filename (str): The filename of the video to load and initialize.

            @return cap (cv2.VideoCapture): The :py:class:`cv2.VideoCapture` object. 
        """
        self.cap = VideoCapture(filename)

        return self.cap

    def select_template(self):
        """Select ROI from first frame of image sequence.

            @return roi : Selected ROI or empty rect if selection canceled.
        """
        ret, self.img = self.cap.read()
        self.gray_img = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        self.roi = cv.selectROI('frame', self.img)
        self.templ = self.gray_img[
            int(self.roi[1]):int(self.roi[1]+self.roi[3]),
            int(self.roi[0]):int(self.roi[0]+self.roi[2])]
        cv.destroyAllWindows()

        return self.templ

    def match_template(self):
        """Match the template image."""
        img_display = self.img.copy()

        method_accepts_mask = (
            cv.TM_SQDIFF == self.match_method or self.match_method == cv.TM_CCORR_NORMED)
        if (self.use_mask and method_accepts_mask):
            result = cv.matchTemplate(
                self.gray_img, self.templ, self.match_method, None, self.mask)
        else:
            result = cv.matchTemplate(
                self.gray_img, self.templ, self.match_method)

        cv.normalize(result, result, 0, 1, cv.NORM_MINMAX, -1)

        _minVal, _maxVal, minLoc, maxLoc = cv.minMaxLoc(result, None)

        if (self.match_method == cv.TM_SQDIFF or self.match_method == cv.TM_SQDIFF_NORMED):
            matchLoc = minLoc
        else:
            matchLoc = maxLoc

        cv.rectangle(img_display, matchLoc,
                     (matchLoc[0] + self.templ.shape[0], matchLoc[1] + self.templ.shape[1]), (255, 0, 0), 2, 8, 0)
        cv.imshow(self.image_window, img_display)
        # cv.rectangle(result, matchLoc, (matchLoc[0] + self.templ.shape[0],
        #              matchLoc[1] + self.templ.shape[1]), (0, 0, 0), 2, 8, 0)
        print(matchLoc)
        cv.imshow(self.templ_window, self.templ)
        cv.imshow(self.templ_window, self.templ)
        cv.imshow(self.diff_window, self.gray_img[
            int(matchLoc[1]):int(matchLoc[1]+self.templ.shape[0]),
            int(matchLoc[0]):int(matchLoc[0]+self.templ.shape[1])])

    def track(self):
        """Track the template image."""
        while self.cap.isOpened():
            ret, self.img = self.cap.read()

            if not ret:
                break

            self.gray_img = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)

            self.match_template()

            cv.waitKey(1)
        return

    def close_tracker(self):
        """Close all windows and release the VideoCapture."""
        self.cap.release()
        cv.destroyAllWindows()
