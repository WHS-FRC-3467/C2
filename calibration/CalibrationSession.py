import datetime
import os
from typing import Sequence, cast

import cv2
import numpy
from cv2.typing import MatLike

from config.ConfigSource import FileConfigSource
from type_defs import FloatArray, IntArray


class CalibrationSession:
    def __init__(self) -> None:
        self._all_charuco_corners: list[FloatArray] = []
        self._all_charuco_ids: list[IntArray] = []
        self._imsize: tuple[int, int] | None = None
        self._aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
        self._aruco_params = cv2.aruco.DetectorParameters()
        self._charuco_board = cv2.aruco.CharucoBoard(
            (12, 9), 0.030, 0.023, self._aruco_dict
        )
        self._detector = cv2.aruco.ArucoDetector(self._aruco_dict, self._aruco_params)
        self._charuco_detector = cv2.aruco.CharucoDetector(self._charuco_board)

    def process_frame(self, image: MatLike, save: bool) -> None:
        image_array = numpy.asarray(image)

        # Get image size
        if self._imsize is None:
            self._imsize = (int(image_array.shape[0]), int(image_array.shape[1]))

        # Detect tags
        corners, _ids, _rejected = self._detector.detectMarkers(image_array)
        if corners:
            _ = cv2.aruco.drawDetectedMarkers(image_array, corners)

            # Find Charuco corners
            charuco_corners, charuco_ids, _, _ = self._charuco_detector.detectBoard(
                image_array
            )
            if (
                charuco_corners is not None
                and charuco_ids is not None
                and len(charuco_corners) > 0
            ):
                _ = cv2.aruco.drawDetectedCornersCharuco(
                    image_array, charuco_corners, charuco_ids
                )

                # Save corners
                if save:
                    self._all_charuco_corners.append(
                        numpy.asarray(charuco_corners, dtype=numpy.float64)
                    )
                    self._all_charuco_ids.append(
                        numpy.asarray(charuco_ids, dtype=numpy.int32)
                    )
                    print("Saved calibration frame")

    def finish(self) -> None:
        if len(self._all_charuco_corners) == 0:
            print("ERROR: No calibration data")
            return

        if os.path.exists(FileConfigSource.CALIBRATION_FILENAME):
            os.remove(FileConfigSource.CALIBRATION_FILENAME)

        if self._imsize is None:
            print("ERROR: Missing image size")
            return

        all_obj_points: list[FloatArray] = []
        all_img_points: list[FloatArray] = []
        for corners, ids in zip(self._all_charuco_corners, self._all_charuco_ids):
            obj_pts, img_pts = self._charuco_board.matchImagePoints(
                cast(Sequence[MatLike], corners),
                ids,
            )
            all_obj_points.append(numpy.asarray(obj_pts, dtype=numpy.float64))
            all_img_points.append(numpy.asarray(img_pts, dtype=numpy.float64))

        retval, camera_matrix, distortion_coefficients, _rvecs, _tvecs = (
            cv2.calibrateCamera(
                all_obj_points,
                all_img_points,
                self._imsize,
                numpy.eye(3, dtype=numpy.float64),
                numpy.zeros((5, 1), dtype=numpy.float64),
            )
        )

        if retval:
            calibration_store = cv2.FileStorage(
                FileConfigSource.CALIBRATION_FILENAME, cv2.FILE_STORAGE_WRITE
            )
            calibration_store.write("calibration_date", str(datetime.datetime.now()))
            calibration_store.write("camera_matrix", camera_matrix)
            calibration_store.write("distortion_coefficients", distortion_coefficients)
            calibration_store.release()
            print("Calibration finished")
        else:
            print("ERROR: Calibration failed")
