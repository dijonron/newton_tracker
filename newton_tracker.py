import matplotlib.pyplot as plt
import cv2 as cv
from cv2 import VideoCapture
import numpy as np
import time


class NewtonTracker:
    """Newton Tracker"""

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
        self.search_area = None
        self.p_0 = [0, 0]
        self.p = [0, 0]

        self.points = np.ndarray((1,2))

    def setMatchMethod(self, match_method: str):
        """Set the search method."""
        self.match_method = match_method

    def addPoint(self, point: np.array):
        """Add tracked point."""
        self.points = np.append(self.points, [point], axis=0)


    def load_video_sequence(self, filename: str):
        """Load and return the (.mp4) video specified by ``filename``.

            @arg filename (str): The filename of the video to load and initialize.

            @return cap (cv2.VideoCapture): The :py:class:`cv2.VideoCapture` object. 
        """
        self.cap = VideoCapture(filename)

        return self.cap

    def select_template(self):
        """Select template from first frame of image sequence."""
        _, self.img = self.cap.read()
        self.gray_img = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        self.roi = cv.selectROI('frame', self.img)
        self.templ = self.gray_img[
            int(self.roi[1]):int(self.roi[1]+self.roi[3]),
            int(self.roi[0]):int(self.roi[0]+self.roi[2])]

        self.search_area = np.add(self.roi, [-20, -20, 40, 40])
        self.p_0 = self.roi[0:2]
        self.p = self.p_0
        self.addPoint(self.p)

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
                self.mask, self.templ, self.match_method)
        cv.normalize(result, result, 0, 1, cv.NORM_MINMAX, -1)
        _minVal, _maxVal, minLoc, maxLoc = cv.minMaxLoc(result, None)

        if (self.match_method == cv.TM_SQDIFF or self.match_method == cv.TM_SQDIFF_NORMED):
            matchLoc = minLoc
        else:
            matchLoc = maxLoc

        u = np.subtract(matchLoc, self.p-self.search_area[0:2])
        self.p = np.add(self.p, u)
        self.addPoint(self.p)

        cv.rectangle(img_display, self.p,
                     (self.p[0] + self.templ.shape[1], self.p[1] + self.templ.shape[0]), (0, 0, 255), 2, 8, 0)
        cv.imshow(self.image_window, img_display)
        
        self.search_area = np.add(self.search_area, np.append(u, [0,0]))

    def track(self):
        """Track the template image."""
        while self.cap.isOpened():
            ret, self.img = self.cap.read()

            if not ret:
                break

            self.gray_img = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)

            self.mask = self.gray_img[int(self.search_area[1]):int(self.search_area[1]+self.search_area[3]),
                                      int(self.search_area[0]):int(self.search_area[0]+self.search_area[2])]
            self.match_template()
            
            cv.waitKey(1)

    def plot(self):
        """Plot tracked points."""
        z = np.polyfit(self.points[1:-1, 0], np.negative(self.points[1:-1, 1]), 2)
        f = np.poly1d(z)
        x_new = np.linspace(0, 1900, 50)
        y_new = f(x_new)

        plt.plot(self.points[1:-1, 0], np.negative(self.points[1:-1, 1]), 'o', x_new, y_new, markersize=3)
        plt.xlim([0, 1960])
        plt.ylim([-1060, 0])
        plt.show()

    def close_tracker(self):
        """Close all windows and release the VideoCapture."""
        self.cap.release()
        cv.destroyAllWindows()
