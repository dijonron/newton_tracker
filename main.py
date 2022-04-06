from re import U
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


def main():
    # colors to track the points
    color = np.random.randint(0, 255, (100, 3))

    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    # load video
    cap = cv.VideoCapture('test.mp4')

    # read first frame and select ROI to track
    ret, prev_frame = cap.read()
    prev_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)
    roi = cv.selectROI("frame", prev_frame)


    input_mask = np.zeros_like(prev_gray, np.uint8)
    input_mask[
        int(roi[1]):int(roi[1]+roi[3]),
        int(roi[0]):int(roi[0]+roi[2])
    ] = prev_gray[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]

    p0 = cv.goodFeaturesToTrack(prev_gray, mask=input_mask, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(prev_frame)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        p1, st, err = cv.calcOpticalFlowPyrLK(
            prev_gray, gray, p0, None, **lk_params)

        # Select good points
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]

        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv.line(mask, (int(a), int(b)),
                           (int(c), int(d)), color[i].tolist(), 2)
            frame = cv.circle(frame, (int(a), int(b)),
                              5, color[i].tolist(), -1)
        img = cv.add(frame, mask)

        cv.imshow('frame', img)

        # Now update the previous frame and previous points
        prev_gray = gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
