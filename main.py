import sys

from lk_tracker import LKTracker
from newton_tracker import NewtonTracker
import cv2 as cv

def main():

    if (len(sys.argv) < 4):
        print('Error: Not enough parameters')
        print('\n\nUsage:\n\tmain.py <video_name> <tracker_name> <method_name>')
        print('\nTracker Options:\n\t\tLK \n\t\tN')
        print('\nMethod Options:\n\t\tTM_SQDIFF \n\t\tTM_SQDIFF_NORMED \n\t\tTM_CCORR \n\t\tTM_CCORR_NORMED')
        return

    match sys.argv[2]:
        case 'LK':
            tracker = LKTracker()
        case 'N':
            tracker = NewtonTracker()
        case _:
            print('Error: Invalid tracker name.')
            print('\n\nUsage:\n\tmain.py <video_name> <tracker_name> <method_name>')
            print('\nTracker Options:\n\t\tLK \n\t\tN')
            print('\nMethod Options:\n\t\tTM_SQDIFF \n\t\tTM_SQDIFF_NORMED \n\t\tTM_CCORR \n\t\tTM_CCORR_NORMED')
            return

    match sys.argv[3]:
        case 'TM_SQDIFF':
            tracker.setMatchMethod(cv.TM_SQDIFF)
        case 'TM_SQDIFF_NORMED':
            tracker.setMatchMethod(cv.TM_SQDIFF_NORMED)
        case 'TM_CCORR':
            tracker.setMatchMethod(cv.TM_CCORR)
        case 'TM_CCORR_NORMED':
            tracker.setMatchMethod(cv.TM_CCORR_NORMED)
        case 'TM_CCOEFF':
            tracker.setMatchMethod(cv.TM_CCOEFF)
        case 'TM_CCOEFF_NORMED':
            tracker.setMatchMethod(cv.TM_CCOEFF_NORMED)
        case _:
            print('Error: Invalid matching method.')
            print('\n\nUsage:\n\tmain.py <video_name> <tracker_name> <method_name>')
            print('\nTracker Options:\n\t\tLK \n\t\tN')
            print('\nMethod Options:\n\t\tTM_SQDIFF \n\t\tTM_SQDIFF_NORMED \n\t\tTM_CCORR \n\t\tTM_CCORR_NORMED')
            return

    tracker.load_video_sequence(sys.argv[1])
    tracker.select_template()
    tracker.track()
    tracker.close_tracker()


if __name__ == "__main__":
    main()
