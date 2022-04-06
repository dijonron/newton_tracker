import cv2 as cv
from cv2 import VideoCapture
import numpy as np


class NewtonTracker:
    """Newton Tracker"""

    def __init__(self):
        # params for ShiTomasi corner detection
        self._feature_params = dict(maxCorners=100,
                                    qualityLevel=0.3,
                                    minDistance=7,
                                    blockSize=7)
        # Parameters for lucas kanade optical flow
        self._lk_params = dict(winSize=(15, 15),
                               maxLevel=2,
                               criteria=(
            cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,
            10,
            0.03))
        self._color = np.random.randint(0, 255, (100, 3))

        self.cap = None  # VideoCapture object
        self.prev_frame = None  # previous frame
        self.prev_gray = None  # grayscale of previous frame
        self.roi = None  # region of interest'
        self.p0 = None  # old points

    def load_video_sequence(self, filename: str):
        """Load and return the (.mp4) video specified by ``filename``.

            @arg filename (str): The filename of the video to load and initialize.

            @return cap (cv2.VideoCapture): The :py:class:`cv2.VideoCapture` object. 
        """
        self.cap = VideoCapture(filename)

        return self.cap

    def select_roi(self):
        """Select ROI from first frame of image sequence.

            @return roi : Selected ROI or empty rect if selection canceled.
        """
        ret, self.prev_frame = self.cap.read()
        self.prev_gray = cv.cvtColor(self.prev_frame, cv.COLOR_BGR2GRAY)
        self.roi = cv.selectROI("frame", self.prev_frame)

        return self.roi

    def get_initial_points(self):
        """Get the initial points to track from the ROI."""
        input_mask = np.zeros_like(self.prev_gray, np.uint8)
        input_mask[
            int(self.roi[1]):int(self.roi[1]+self.roi[3]),
            int(self.roi[0]):int(self.roi[0]+self.roi[2])
        ] = self.prev_gray[int(self.roi[1]):int(self.roi[1]+self.roi[3]), int(self.roi[0]):int(self.roi[0]+self.roi[2])]

        self.p0 = cv.goodFeaturesToTrack(
            self.prev_gray, mask=input_mask, **self._feature_params)

        return self.p0

    def track(self):
        """track"""
        # Create a mask image for drawing purposes
        mask = np.zeros_like(self.prev_frame)

        while self.cap.isOpened():
            ret, frame = self.cap.read()

            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            p1, st, err = cv.calcOpticalFlowPyrLK(
                self.prev_gray, gray, self.p0, None, **self._lk_params)

            # Select good points
            if p1 is not None:
                good_new = p1[st == 1]
                good_old = self.p0[st == 1]

            # draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv.line(mask,
                               (int(a), int(b)),
                               (int(c), int(d)),
                               self._color[i].tolist(),
                               2)
                frame = cv.circle(frame,
                                  (int(a), int(b)),
                                  5,
                                  self._color[i].tolist(),
                                  -1)
            img = cv.add(frame, mask)

            cv.imshow('frame', img)

            # Now update the previous frame and previous points
            self.prev_gray = gray.copy()
            self.p0 = good_new.reshape(-1, 1, 2)

            if cv.waitKey(1) == ord('q'):
                break

    def close_tracker(self):
        """Close all windows and release the VideoCapture."""
        self.cap.release()
        cv.destroyAllWindows()