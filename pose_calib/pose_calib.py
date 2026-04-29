#!/usr/bin/env python3

"""
This file is part of the "Pose Calib" project.
It is subject to the license terms in the LICENSE file found
in the top-level directory of this distribution.

@author Pavel Rojtberg
"""

import cv2
import argparse
import sys
import requests
import threading

from ui import UserGuidance
from utils import ChArucoDetector

import numpy as np


class MJPEGCapture:
    def __init__(self, url):
        self._url = url
        self._frame = None
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        while self._running:
            try:
                resp = requests.get(self._url, stream=True, timeout=5)
                buf = b''
                for chunk in resp.iter_content(chunk_size=4096):
                    if not self._running:
                        break
                    buf += chunk
                    # Look for JPEG start and end markers
                    start = buf.find(b'\xff\xd8')
                    end = buf.find(b'\xff\xd9')
                    if start != -1 and end != -1 and end > start:
                        jpg = buf[start:end + 2]
                        buf = buf[end + 2:]
                        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if frame is not None:
                            with self._lock:
                                self._frame = frame
            except Exception as e:
                print(f"MJPEG stream error: {e}, reconnecting...")
                import time
                time.sleep(1)

    def read(self):
        with self._lock:
            if self._frame is not None:
                return True, self._frame.copy()
            return False, None

    def isOpened(self):
        return self._running

    def release(self):
        self._running = False
    
def main():
    parser = argparse.ArgumentParser(description="Interactive camera calibration using efficient pose selection")
    parser.add_argument("-c", "--config", help="path to calibration configuration (e.g. data/calib_config.yml)")
    parser.add_argument("-o", "--outfile", help="path to calibration output (defaults to calib_<cameraId>.yml)")
    parser.add_argument("-m", "--mirror", action="store_true", help="horizontally flip the camera image for display")
    parser.add_argument("--address", required=True, help="MJPEG server address (hostname or IP)")
    parser.add_argument("--port", type=int, required=True, help="MJPEG server port")
    parser.add_argument("--name", required=True, help="camera name (used for output filename)")
    args = parser.parse_args()

    if args.config is None:
        print("falling back to "+sys.path[0]+"/data/calib_config.yml")
        args.config = sys.path[0]+"/data/calib_config.yml"

    cfg = cv2.FileStorage(args.config, cv2.FILE_STORAGE_READ)
    assert cfg.isOpened()

    calib_name = args.name

    # Video I/O
    live = cfg.getNode("images").empty()
    if live:
        stream_url = "http://{}:{}/stream".format(args.address, args.port)
        print("Connecting to MJPEG stream at {}...".format(stream_url))
        cap = MJPEGCapture(stream_url)
        cv2.namedWindow("PoseCalib", cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
        wait = 1
    else:
        cv2.namedWindow("PoseCalib")
        cap = cv2.VideoCapture(cfg.getNode("images").string() + "frame%0d.png", cv2.CAP_IMAGES)
        wait = 0
        assert cap.isOpened()

    tracker = ChArucoDetector(cfg)


    # user guidance
    ugui = UserGuidance(tracker, cfg.getNode("terminate_var").real())

    # runtime variables
    img = None
    mirror = False
    save = False

    while True:
        force = not live  # force add frame to calibration

        status, _img = cap.read()
        if status:
            img = _img
        else:
            force = False
            if img is None:
                continue

        tracker.detect(img)

        if save:
            save = False
            force = True

        out = img.copy()

        ugui.draw(out, mirror)

        ugui.update(force)
        
        if ugui.converged:
            if args.outfile is None:
                outfile = "calib_{}.yml".format(calib_name)
            else:
                outfile = args.outfile
            ugui.write(outfile)
            print("Calibration complete, written to {}".format(outfile))
            break

        if ugui.user_info_text:
            # Draw info text directly on the output image
            for i, line in enumerate(ugui.user_info_text.split('\n')):
                cv2.putText(out, line, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("PoseCalib", out)
        k = cv2.waitKey(wait)

        if k == 27:
            break
        elif k == ord('m'):
            mirror = not mirror
        elif k == ord('c'):
            save = True

if __name__ == "__main__":
    main()
