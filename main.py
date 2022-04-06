from newton_tracker import NewtonTracker


def main():
    tracker = NewtonTracker()

    tracker.load_video_sequence('test.mp4')
    tracker.select_roi()
    tracker.get_initial_points()

    tracker.track()

    tracker.close_tracker()


if __name__ == "__main__":
    main()
