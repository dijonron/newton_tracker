import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from cv2 import VideoCapture


class ProjectileTracker:
    """Projectile Tracker

    Template matching based tracker the uses the objects trajectory to predict its next 
    state update when the confidence of match is below a threshold value.

    Attributes:
    ----------
        cap: A :py:class:`cv2.VideoCapture` object. Holds the video sequence being analysed. 

        img: A :py:class:`numpy.ndarray` representing the current image frame of the video sequence from :py:class:`ProjectileTracker.cap`.

        gray_img: A :py:class:`numpy.ndarray` representing a grayscale copy of :py:class:`ProjectileTracker.img`.

        roi: A :py:class:`tuple` represneting the selected roi.

        templ: A :py:class:`numpy.ndarray` representing the template image that is being tracked.

        search_area: A :py:class:`numpy.ndarray` representing the search area that the tracker is looking to match the template in.

        match_method: The :py:calss:`cv.TemplateMatchModes` being used to match the template to the image.

        p_0: A :py:class:`numpy.ndarray` containing the initial template match position.

        p: A :py:class:`numpy.ndarray` containing the latest template match position.

        points: A :py:class:`numpy.ndarray` containing all of the tracked points from the sequence.

        plot: A :py:type:`bool` flag for plotting final trajectory or not.

        delay (:py:type:`int`) The delay in milliseconds between frames.

        frame: An :py:class:`int` representing the current frame of the sequence. 

        thresh: The confidence threshold for template matching.
    """

    def __init__(self, plot: bool = False, delay: int = 1, thresh: float = 0.97):
        """Inits the tracker.
        
        @arg plot (:py:type:`bool`): Boolean flag for plotting final trajectory.

        @arg delay (:py:type:`int`) Delay in milliseconds. 0 is the special value that means "forever".
        
        """
        self.use_mask = False
        self.mask = None
        self.image_window = "Source Image"
        self.templ_window = "Template Window"
        self.diff_window = "Diff Window"

        self.cap = None
        self.img = None
        self.gray_img = None
        self.roi = None
        self.templ = None
        self.search_area = None
        self.match_method = None
        self.p_0 = np.array([0, 0])
        self.p = np.array([0, 0])
        self.points = np.ndarray((1, 2))
        self.predictions = np.array([0])

        self.plot = plot
        self.delay = delay
        self.frame = 0
        self.thresh = thresh

    def set_match_method(self, match_method: str):
        """Set the search method.

        @arg match_method (:py:type:`str`): String represneting the match method to be used.
        """
        self.match_method = match_method

    def add_point(self, point: np.array):
        """Add point to the `self.points` array.

        @arg point (:py:class:`np.array`): Point to be added to `self.points`
        """
        self.points = np.append(self.points, [point], axis=0)

    def load_video_sequence(self, filename: str):
        """Load and return the (.mp4) video specified by ``filename``.

        @arg filename (:py:type:`str`): The filename of the video to load and initialize.

        @return cap (:py:class:`cv2.VideoCapture`): The :py:class:`cv2.VideoCapture` object. 
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
        self.add_point(self.p)
        self.predictions = np.append(self.predictions, 0)

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
        # cv.normalize(result, result, 1, 0, cv.NORM_MINMAX, -1)
        minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(result, None)

        if (self.match_method == cv.TM_SQDIFF or self.match_method == cv.TM_SQDIFF_NORMED):
            matchConf = 1-minVal
        else:
            matchConf = maxVal

        if (abs(matchConf) < self.thresh):
            self.predict_update()
            cv.rectangle(img_display, self.p,
                         (self.p[0] + self.templ.shape[1], self.p[1] + self.templ.shape[0]), (0, 0, 255), 2, 8, 0)
            cv.imshow(self.image_window, img_display)
            u = self.points[-1] - self.points[-2]
            u = u.astype(int)
            self.search_area = np.add(self.search_area, np.append(u, [0, 0]))
            return

        if (self.match_method == cv.TM_SQDIFF or self.match_method == cv.TM_SQDIFF_NORMED):
            matchLoc = minLoc
        else:
            matchLoc = maxLoc

        u = np.subtract(matchLoc, self.p-self.search_area[0:2])
        self.p = np.add(self.p, u)

        self.add_point(self.p)
        self.predictions = np.append(self.predictions, 0)
        cv.rectangle(img_display, self.p,
                     (self.p[0] + self.templ.shape[1], self.p[1] + self.templ.shape[0]), (0, 255, 0), 2, 8, 0)
        cv.imshow(self.image_window, img_display)

        self.search_area = np.add(self.search_area, np.append(u, [0, 0]))

    def track(self):
        """Track the template image.

        This is the main function of the tracker. It will read all the frames of the video sequence, and call
        :py:method:`ProjectileTracker.match_template()` to update the state of the tracker.
        """
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
            cv.waitKey(self.delay)

    def predict_update(self):
        """ Predict the next location of the ROI based on the calculated trajectory of the object."""
        velocity = self.get_velocity()
        x_prediction = self.points[-1][0] + velocity
        z = np.polyfit(self.points[1:-1, 0], self.points[1:-1, 1], 2)
        f = np.poly1d(z)
        y_prediction = f(x_prediction)
        self.p = [int(np.ceil(x_prediction)), int(np.ceil(y_prediction))]
        self.add_point(self.p)
        self.predictions = np.append(self.predictions, 1)

    def plot_trajectory(self):
        """Plot tracked points."""
        z = np.polyfit(self.points[1:-1, 0],
                       np.negative(self.points[1:-1, 1]), 2)
        f = np.poly1d(z)
        x_new = np.linspace(0, 1900, 50)
        y_new = f(x_new)
        measured_points = self.points[self.predictions == 0]
        predicted_points = self.points[self.predictions == 1]
        # np.savetxt("prediction_measured.csv", measured_points, delimiter=",")
        # np.savetxt("prediction_predicted.csv", predicted_points, delimiter=",")

        plt.plot(x_new, y_new, '--')
        plt.plot(measured_points[1:-1, 0], np.negative(measured_points[1:-1, 1]),
                 'go', markersize=3)
        plt.plot(predicted_points[1:-1, 0], np.negative(predicted_points[1:-1, 1]),
                 'rx', markersize=3)

        plt.xlim([0, 1960])
        plt.ylim([-1060, 0])
        plt.legend(['Calculate Trajectory', 'Measured Points', 'Predicted Points'])
        plt.show()

    def get_velocity(self) -> float:
        """Return the x velocity of the tracked object [pixels/frame].

        @return velocity (float): The x velocity
        """
        dx = self.points[-1][0] - self.points[1][0]
        return dx/self.frame

    def get_initial_params(self):
        """Get initial params of motion.

        @return y0: The y pixel coord of the initial match 

        @return v0: The initial velocity estimation

        @return alpha: The initial angle estimation
        """
        y0 = self.points[1][1]
        v0 = (self.points[3] - self.points[1])/2
        alpha = np.arctan(-v0[1]/v0[0])

        return y0, v0, alpha

    def close_tracker(self):
        """Close all windows and release the VideoCapture."""
        self.cap.release()
        cv.destroyAllWindows()

        if self.plot:
            self.plot_trajectory()
