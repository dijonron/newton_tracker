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
        self.match_method = None
        self.frame = 0
        self.thresh = 3e-9

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
        self.frame += 1
        # self.gray_img = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        self.gray_img = self.img
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
        # cv.imshow(self.diff_window, self.gray_img[self.p[1]:self.p[1] + self.templ.shape[0], self.p[0]: self.p[0] + self.templ.shape[1]] - self.templ)
        # cv.waitKey(0)
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
        minVal, _maxVal, minLoc, maxLoc = cv.minMaxLoc(result, None)

        if (abs(minVal) > self.thresh):
            self.predictUpdate()
            cv.rectangle(img_display, self.p,
                     (self.p[0] + self.templ.shape[1], self.p[1] + self.templ.shape[0]), (0, 0, 255), 2, 8, 0)
            cv.imshow(self.image_window, img_display)
            u = self.points[-1] - self.points[-2]
            u = u.astype(int)
            self.search_area = np.add(self.search_area, np.append(u, [0,0]))
            return  

        if (self.match_method == cv.TM_SQDIFF or self.match_method == cv.TM_SQDIFF_NORMED):
            matchLoc = minLoc
        else:
            matchLoc = maxLoc

        u = np.subtract(matchLoc, self.p-self.search_area[0:2])
        self.p = np.add(self.p, u)

        self.addPoint(self.p)
        cv.rectangle(img_display, self.p,
                     (self.p[0] + self.templ.shape[1], self.p[1] + self.templ.shape[0]), (0, 255, 0), 2, 8, 0)
        cv.imshow(self.image_window, img_display)
        # cv.imshow('mask', self.mask)
        
        self.search_area = np.add(self.search_area, np.append(u, [0,0]))

    def track(self):
        """Track the template image."""
        while self.cap.isOpened():
            ret, self.img = self.cap.read()
            self.frame += 1
            if not ret:
                break

            # self.gray_img = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
            self.gray_img = self.img

            self.mask = self.gray_img[int(self.search_area[1]):int(self.search_area[1]+self.search_area[3]),
                                      int(self.search_area[0]):int(self.search_area[0]+self.search_area[2])]
            self.match_template()
            
            cv.waitKey(1)

    def predictUpdate(self):
        velocity = self.getVelocity()
        x_prediction = self.points[-1][0] + velocity
        z = np.polyfit(self.points[1:-1, 0], self.points[1:-1, 1], 2)
        f = np.poly1d(z)
        y_prediction = f(x_prediction)
        self.p = [int(np.ceil(x_prediction)), int(np.ceil(y_prediction))]
        self.addPoint(self.p)

    def plot(self, predict=False, *args):
        """Plot tracked points."""
        z = np.polyfit(self.points[1:-1, 0], np.negative(self.points[1:-1, 1]), 2)
        f = np.poly1d(z)
        print(f)
        x_new = np.linspace(0, 1900, 50)
        y_new = f(x_new)

        plt.plot(self.points[1:-1, 0], np.negative(self.points[1:-1, 1]), 'o', x_new, y_new, markersize=3)

        if predict:
            velocity = self.getVelocity()
            x_prediction = self.points[-1][0] + velocity
            plt.plot(x_prediction, f(x_prediction), 'rx')

        plt.xlim([0, 1960])
        plt.ylim([-1060, 0])
        plt.show()

    def getVelocity(self) -> float:
        """Return the x velocity of the tracked object [pixels/frame].
            
        @return velocity (float): The x velocity
        """
        dx = self.points[-1][0] - self.points[1][0]
        return dx/self.frame

    def getInitialParams(self):
        """Get initial params of motion."""
        y0 = self.points[1][1]
        v0 = (self.points[3] - self.points[1])/2
        alpha = np.arctan(-v0[1]/v0[0])
        print(v0, y0, alpha)
        return y0, v0, alpha

    def close_tracker(self):
        """Close all windows and release the VideoCapture."""
        self.cap.release()
        cv.destroyAllWindows()
