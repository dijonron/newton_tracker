# Newton Tracker

Adding Newton's Laws to predict and update state of a registration tracker for an occluded object.

Work complete for the final project portion of CMPUT 428 - Computer Vision, University of Alberta, Winter 2022.

# Usage

```
main.py -f <file> -t <tracker> -m <method> -p

        -f --file       The filename of the video to track. Must be .mp4.

        -t --tracker    The type of tracker to use.

                        Options:
                                LK      Lucas Kanade Optical Flow Point Tracker.
                                T       Template matching tracker.
                                N       Template matching tracker, using Newton's Law's to update occluded state.

        -m --method     Method of matching template image to current frame.
                        Only needed for Template or Newton Tracker.

                        Options:
                                TM_SQDIFF
                                TM_SQDIFF_NORMED
                                TM_CCORR
                                TM_CCORR_NORMED

        -p --plot       Plot the final trajectory of the object.

        -d --delay      Delay, in milliseconds, between showing next frame.
```

# Contributors

- Dalton Ronan
