import time

import cv2
import numpy as np

# from realsense_camera import MultiCamera
from deoxys.camera.orbbec_camera import MultiCamera

if __name__ == "__main__":
    multi_camera = MultiCamera()
    camera_ids = list(multi_camera.cameras.keys())
    print("camera_ids", camera_ids)

    while True:
        # Get img info
        images = multi_camera.get_frame()
        rgbs = []
        for camera_id in camera_ids:
            rgb, depth = images[camera_id]
            rgbs.append(rgb)

        cv2.imshow("rgb", np.hstack(rgbs))
        cv2.waitKey(1)

        time.sleep(0.1)
