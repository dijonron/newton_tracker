import sys

from lk_tracker import LKTracker
from newton_tracker import NewtonTracker


def main():

    match sys.argv[1]:
        case 'LK':
            tracker = LKTracker()
        case 'N':
            tracker = NewtonTracker()
        case _:
            print('Provide a valid tracking mode.')
            return

    if sys.argv[2] is None:
        print('Error: Provide an input video file name!')
        return

    tracker.load_video_sequence(sys.argv[2])
    tracker.select_roi()
    tracker.get_initial_points()
    tracker.track()
    tracker.close_tracker()


if __name__ == "__main__":
    main()
