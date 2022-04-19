import sys
import getopt

from lk_tracker import LKTracker
from newton_tracker import NewtonTracker
import cv2 as cv

def print_help():
    print('main.py -f <file> -t <tracker> -m <method> -p')
    print('\n\t-f --file\tThe filename of the video to track. Must be .mp4.')

    print('\n\t-t --tracker\tThe type of tracker to use.')
    print('\n\t\t\tOptions:\n\t\t\t\tLK\tLucas Kanade Optical Flow Point Tracker.\n\t\t\t\tN\tTemplate matching tracker, using Newton\'s Law\'s to update occluded state.')

    print(
        '\n\t-m --method\tMethod of matching template image to current frame. Only needed for Newton Tracker.')
    print(
        '\n\t\t\tOptions:\n\t\t\t\tTM_SQDIFF \n\t\t\t\tTM_SQDIFF_NORMED \n\t\t\t\tTM_CCORR \n\t\t\t\tTM_CCORR_NORMED')

def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hf:t:m:p', [
                                   'file=', 'tracker=', 'method=', 'plot'])
    except getopt.GetoptError:
        print_help()
        sys.exit(2)

    for opt, arg in opts:
        match opt:
            case '-h':
                print_help()
                sys.exit(2)
            case '-f' | '--file':
                filename = arg
            case '-t' | '--tracker':
                tracker_arg = arg
            case '-m' | '--match-method':
                match_method = arg
            case '-p' | '--plot':
                plot = arg

    match tracker_arg:
        case 'LK':
            tracker = LKTracker()
        case 'N':
            tracker = NewtonTracker()
        case _:
            print('Error: Invalid tracker name.')
            print('\n\nUsage:\n\tmain.py <video_name> <tracker_name> <method_name>')
            print('\nTracker Options:\n\t\t\tLK \n\t\tN')
            print(
                '\nMethod Options:\n\t\tTM_SQDIFF \n\t\tTM_SQDIFF_NORMED \n\t\tTM_CCORR \n\t\tTM_CCORR_NORMED')
            return

    if tracker_arg == 'N':
        match match_method:
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
                print(
                    '\nMethod Options:\n\t\tTM_SQDIFF \n\t\tTM_SQDIFF_NORMED \n\t\tTM_CCORR \n\t\tTM_CCORR_NORMED')
                return

    tracker.load_video_sequence(filename)
    tracker.select_template()
    tracker.track()
    tracker.close_tracker()


if __name__ == '__main__':
    main()
