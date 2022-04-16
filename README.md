# Newton Tracker

Adding Newton's Laws to predict and update state of a registration tracker for an occluded object.

Work complete for the final project portion of CMPUT 428 - Computer Vision, University of Alberta, Winter 2022.

# Usage

```
main.py <video_name> <tracker_name> <method_name>

Tracker Options:
        LK      Lukas Kanade    - Basic open CV LK optical flow tracker
        N       Newton Tracker  - Template matching tracker using Newton's Laws to update search area when tracking lost

Method Options:
        TM_SQDIFF               - Sum of Square Differences
        TM_SQDIFF_NORMED        - Not supported
        TM_CCORR                - Not supported
        TM_CCORR_NORMED         - Not supported
```

# Contributors

- Dalton Ronan
