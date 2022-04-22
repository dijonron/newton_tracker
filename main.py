import getopt
import sys
import time
import cv2 as cv

from lk_tracker import LKTracker
from projectile_tracker import ProjectileTracker
from template_tracker import TemplateTracker


def print_help():
    print('main.py -f <file> -t <tracker> -m <method> -p')
    print('\n\t-f --file\tThe filename of the video to track. Must be .mp4.')

    print('\n\t-t --tracker\tThe type of tracker to use.')
    print('\n\t\t\tOptions:\n\t\t\t\tLK\tLucas Kanade Optical Flow Point Tracker.\
    \n\t\t\t\tT\tTemplate matching tracker.\
    \n\t\t\t\tP\T matching tracker, using interpolated parabola to update occluded state.\
    \n\t\t\t\tN\tTemplate matching tracker, using Newton\'s Law\'s to update occluded state.')

    print(
        '\n\t-m --method\tMethod of matching template image to current frame. \
        \n\t\t\tOnly needed for Template or Newton Tracker.')
    print(
        '\n\t\t\tOptions:\n\t\t\t\tTM_SQDIFF \n\t\t\t\tTM_SQDIFF_NORMED \n\t\t\t\tTM_CCORR \n\t\t\t\tTM_CCORR_NORMED')

    print('\n\t-p --plot\tPlot the final trajectory of the object.')

    print('\n\t-d --delay\tDelay, in milliseconds, between showing next frame.')

def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hf:t:m:pd:', [
                                   'file=', 'tracker=', 'method=', 'plot', 'delay='])
    except getopt.GetoptError:
        print_help()
        sys.exit(2)
    plot = False
    delay = 1
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
                plot = True
            case '-d' | '--delay':
                delay = int(arg)

    match tracker_arg:
        case 'LK':
            tracker = LKTracker()
        case 'T':
            tracker = TemplateTracker(plot, delay)
        case 'P':
            tracker = ProjectileTracker(plot, delay)
        case _:
            print_help()
            sys.exit(2)

    if tracker_arg in('N', 'P', 'T'):
        match match_method:
            case 'TM_SQDIFF':
                tracker.set_match_method(cv.TM_SQDIFF)
            case 'TM_SQDIFF_NORMED':
                tracker.set_match_method(cv.TM_SQDIFF_NORMED)
            case 'TM_CCORR':
                tracker.set_match_method(cv.TM_CCORR)
            case 'TM_CCORR_NORMED':
                tracker.set_match_method(cv.TM_CCORR_NORMED)
            case 'TM_CCOEFF':
                tracker.set_match_method(cv.TM_CCOEFF)
            case 'TM_CCOEFF_NORMED':
                tracker.set_match_method(cv.TM_CCOEFF_NORMED)
            case _:
                print_help()
                sys.exit(2)

    tracker.load_video_sequence(filename)
    tracker.select_template()
    # start_time = time.time()
    tracker.track()
    tracker.close_tracker()
    # print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()
