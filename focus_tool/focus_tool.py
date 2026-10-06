#!/usr/bin/env python3

import argparse
import threading
import time
from collections import deque

import cv2
import numpy as np
import requests


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
                buf = b""
                for chunk in resp.iter_content(chunk_size=4096):
                    if not self._running:
                        break
                    buf += chunk
                    start = buf.find(b"\xff\xd8")
                    end = buf.find(b"\xff\xd9")
                    if start != -1 and end != -1 and end > start:
                        jpg = buf[start : end + 2]
                        buf = buf[end + 2 :]
                        frame = cv2.imdecode(
                            np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR
                        )
                        if frame is not None:
                            with self._lock:
                                self._frame = frame
            except Exception as e:
                print(f"MJPEG stream error: {e}, reconnecting...")
                time.sleep(1)

    def read(self):
        with self._lock:
            if self._frame is not None:
                return True, self._frame.copy()
            return False, None

    def release(self):
        self._running = False


def laplacian_variance(gray):
    """Variance of Laplacian - higher means sharper."""
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def region_sharpness(gray, rows, cols):
    """Compute sharpness for a grid of regions."""
    h, w = gray.shape
    rh, rw = h // rows, w // cols
    values = np.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            roi = gray[r * rh : (r + 1) * rh, c * rw : (c + 1) * rw]
            values[r, c] = laplacian_variance(roi)
    return values, rh, rw


def draw_slider(frame, pct, x, y, w, h):
    """Draw a green/red go/no-go slider bar."""
    # Background bar - gradient from red to green
    for i in range(w):
        t = i / w
        if t < 0.5:
            color = (0, int(255 * t * 2), int(255 * (1 - t * 2)))  # red to yellow
        else:
            color = (0, 255, int(255 * (1 - t) * 2))  # yellow to green
        cv2.line(frame, (x + i, y), (x + i, y + h), color, 1)

    # Border
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)

    # Indicator needle
    needle_x = x + int(pct / 100.0 * w)
    needle_x = max(x, min(x + w, needle_x))
    cv2.line(frame, (needle_x, y - 5), (needle_x, y + h + 5), (255, 255, 255), 3)
    cv2.line(frame, (needle_x, y - 5), (needle_x, y + h + 5), (0, 0, 0), 1)


def main():
    parser = argparse.ArgumentParser(description="Camera focus tool using MJPEG stream")
    parser.add_argument("--address", required=True, help="MJPEG server address")
    parser.add_argument("--port", type=int, required=True, help="MJPEG server port")
    args = parser.parse_args()

    stream_url = "http://{}:{}/stream".format(args.address, args.port)
    print("Connecting to MJPEG stream at {}...".format(stream_url))
    cap = MJPEGCapture(stream_url)

    cv2.namedWindow("Focus Tool", cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)

    # Use a rolling window to track best, so transient spikes don't stick
    history = deque(maxlen=100)
    GRID_ROWS, GRID_COLS = 3, 4
    ZOOM_SIZE = 200  # half-size of crop region in original pixels
    ZOOM_SCALE = 2  # magnification factor
    zoom_center = None  # (x, y) or None

    def on_mouse(event, x, y, flags, param):
        nonlocal zoom_center
        if event == cv2.EVENT_LBUTTONDOWN:
            zoom_center = (x, y)

    cv2.setMouseCallback("Focus Tool", on_mouse)

    while True:
        ret, frame = cap.read()
        if not ret:
            if cv2.waitKey(1) == 27:
                break
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness = laplacian_variance(gray)
        history.append(sharpness)

        # Use 95th percentile of recent history as "best" to ignore outlier spikes
        best_sharpness = (
            np.percentile(list(history), 95) if len(history) >= 5 else sharpness
        )
        best_sharpness = max(best_sharpness, 1.0)
        pct = min(sharpness / best_sharpness * 100, 100.0)

        # Region sharpness grid
        grid_vals, rh, rw = region_sharpness(gray, GRID_ROWS, GRID_COLS)
        grid_max = grid_vals.max() if grid_vals.max() > 0 else 1.0

        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                val = grid_vals[r, c]
                cx = c * rw + rw // 2
                cy = r * rh + rh // 2

                # Color based on relative sharpness within the grid
                rel = val / grid_max
                if rel > 0.7:
                    color = (0, 255, 0)
                elif rel > 0.4:
                    color = (0, 255, 255)
                else:
                    color = (0, 0, 255)

                label = "{:.0f}".format(val)
                text_sz = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                tx = cx - text_sz[0] // 2
                ty = cy + text_sz[1] // 2

                # Background for readability
                cv2.putText(
                    frame,
                    label,
                    (tx + 1, ty + 1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    3,
                )
                cv2.putText(
                    frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                )

        # Draw overall sharpness + slider at top
        frame_h, frame_w = frame.shape[:2]
        bar_y = 10
        bar_h = 30
        bar_margin = 10
        slider_w = frame_w - 20

        # Semi-transparent background for top bar area
        overlay = frame[0 : bar_y + bar_h + 45, :].copy()
        cv2.rectangle(overlay, (0, 0), (frame_w, bar_y + bar_h + 45), (0, 0, 0), -1)
        cv2.addWeighted(
            overlay,
            0.5,
            frame[0 : bar_y + bar_h + 45, :],
            0.5,
            0,
            frame[0 : bar_y + bar_h + 45, :],
        )

        draw_slider(frame, pct, bar_margin, bar_y, slider_w, bar_h)

        # Text below slider
        text_y = bar_y + bar_h + 20
        status = "GOOD" if pct > 90 else "OK" if pct > 70 else "POOR"
        status_color = (
            (0, 255, 0) if pct > 90 else (0, 255, 255) if pct > 70 else (0, 0, 255)
        )

        cv2.putText(
            frame,
            "Focus: {:.0f}  Best: {:.0f}  ({:.0f}%)".format(
                sharpness, best_sharpness, pct
            ),
            (bar_margin, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            status,
            (frame_w - 100, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            status_color,
            2,
        )

        # Draw zoom region indicator on main frame
        if zoom_center is not None:
            zx, zy = zoom_center
            x1 = max(0, zx - ZOOM_SIZE)
            y1 = max(0, zy - ZOOM_SIZE)
            x2 = min(frame_w, zx + ZOOM_SIZE)
            y2 = min(frame_h, zy + ZOOM_SIZE)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

            # Extract crop from the raw frame (before HUD was drawn) and zoom in
            crop = frame[y1:y2, x1:x2]
            zoomed = cv2.resize(
                crop,
                (crop.shape[1] * ZOOM_SCALE, crop.shape[0] * ZOOM_SCALE),
                interpolation=cv2.INTER_LINEAR,
            )

            # Show sharpness of the zoomed region
            zoom_gray = gray[y1:y2, x1:x2]
            zoom_sharp = laplacian_variance(zoom_gray)
            cv2.putText(
                zoomed,
                "Sharpness: {:.0f}".format(zoom_sharp),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            cv2.imshow("Zoom", zoomed)

        cv2.imshow("Focus Tool", frame)
        k = cv2.waitKey(1)
        if k == 27:
            break
        elif k == ord("z"):
            zoom_center = None
            cv2.destroyWindow("Zoom")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
