"""Standalone NetworkTables server that publishes config and displays pose results."""

import json
import math
import os
import sys
import time
from typing import Any

import ntcore

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "schema"))

from dsv0.Frame import Frame  # type: ignore[import-not-found]
from dsv0.Vec3 import Vec3  # type: ignore[import-not-found]
from dsv0.Quaternion import Quaternion  # type: ignore[import-not-found]

DEVICE_ID = "dsv0"
NUM_CAMERAS = 4

# Minimal tag layout with a single tag for testing
TAG_LAYOUT = json.dumps(
    {
        "tags": [
            {
                "ID": 18,
                "pose": {
                    "translation": {"x": 0.0, "y": 0.0, "z": 1.0},
                    "rotation": {
                        "quaternion": {"W": 1.0, "X": 0.0, "Y": 0.0, "Z": 0.0}
                    },
                },
            }
        ]
    }
)


def parse_frame(buf: bytes) -> list[dict[str, Any] | None] | None:
    """Parse a flatbuffer Frame from raw bytes. Returns list of per-camera results."""
    if not buf or len(buf) == 0:
        return None

    frame = Frame.GetRootAsFrame(buf, 0)
    results: list[dict[str, Any] | None] = []
    for i in range(frame.CamerasLength()):
        cam = frame.Cameras(i)
        cam_obs = cam.CameraObservation()
        if cam_obs is None:
            results.append(None)
            continue

        sol = cam_obs.Solution0()
        if sol is None:
            results.append(None)
            continue

        v = Vec3()
        sol.Pose().Translation(v)
        q = Quaternion()
        sol.Pose().Rotation(q)

        tag_ids = [cam_obs.TagIds(j) for j in range(cam_obs.TagIdsLength())]

        entry = {
            "camera_index": cam.CameraIndex(),
            "translation": {"x": v.X(), "y": v.Y(), "z": v.Z()},
            "rotation": {"qw": q.W(), "qx": q.X(), "qy": q.Y(), "qz": q.Z()},
            "error": sol.Error(),
            "tag_ids": tag_ids,
            "fps": cam.Fps(),
        }

        # Check for alternate solution
        sol1 = cam_obs.Solution1()
        if sol1 is not None:
            entry["alt_error"] = sol1.Error()

        results.append(entry)
    return results


def format_result(r: dict[str, Any] | None) -> str:
    """Format a parsed camera result into a compact readable string."""
    if r is None:
        return "no detection"

    t = r["translation"]
    q = r["rotation"]
    yaw = math.degrees(
        math.atan2(
            2 * (q["qw"] * q["qz"] + q["qx"] * q["qy"]),
            1 - 2 * (q["qy"] ** 2 + q["qz"] ** 2),
        )
    )

    s = f"tags={r['tag_ids']} X={t['x']:+.3f} Y={t['y']:+.3f} Z={t['z']:+.3f} yaw={yaw:+.1f}° err={r['error']:.4f}"
    if "alt_error" in r:
        s += f" | alt_err={r['alt_error']:.4f}"
    return s


if __name__ == "__main__":
    inst = ntcore.NetworkTableInstance.getDefault()
    inst.startServer()
    print("NetworkTables server started")

    # Config publishers
    table = inst.getTable(f"/{DEVICE_ID}/config")
    camera_id_pub = table.getStringTopic("camera_id").publish()
    res_w_pub = table.getIntegerTopic("camera_resolution_width").publish()
    res_h_pub = table.getIntegerTopic("camera_resolution_height").publish()
    exp_pub = table.getIntegerTopic("camera_exposure").publish()
    gain_pub = table.getIntegerTopic("camera_gain").publish()
    fid_pub = table.getDoubleTopic("fiducial_size_m").publish()
    layout_pub = table.getStringTopic("tag_layout").publish()

    def publish_config():
        camera_id_pub.set("0")
        res_w_pub.set(1280)
        res_h_pub.set(800)
        exp_pub.set(100)
        gain_pub.set(0)
        fid_pub.set(0.1651)
        layout_pub.set(TAG_LAYOUT)

    publish_config()

    # Per-camera flatbuffer observation subscribers
    obs_subs = []
    for i in range(NUM_CAMERAS):
        cam_table = inst.getTable(f"/{DEVICE_ID}/output/camera_{i}")
        obs_subs.append(
            cam_table.getRawTopic("observation").subscribe("dsv0_fb", bytes())
        )

    # Per-camera pose publishers (structured strings for easy viewing)
    pose_table = inst.getTable(f"/{DEVICE_ID}/poses")
    pose_pubs = [
        pose_table.getStringTopic(f"camera_{i}").publish() for i in range(NUM_CAMERAS)
    ]

    fps_sub = (
        inst.getTable(f"/{DEVICE_ID}/output/camera_0")
        .getIntegerTopic("fps")
        .subscribe(0)
    )

    print(f"Published config to /{DEVICE_ID}/config")
    print(f"Subscribing to {NUM_CAMERAS} camera outputs (flatbuffer)")
    print(f"Republishing poses to /{DEVICE_ID}/poses/")
    print("Press Ctrl+C to stop\n")

    last_connections = 0
    while True:
        connections = inst.getConnections()
        if len(connections) != last_connections:
            last_connections = len(connections)
            print(f"Connections: {last_connections}")
            for c in connections:
                print(f"  {c.remote_id} @ {c.remote_ip}")
            publish_config()

        any_detection = False
        for i in range(NUM_CAMERAS):
            buf = obs_subs[i].get()
            if not buf or len(buf) == 0:
                pose_pubs[i].set("no detection")
                continue

            results = parse_frame(buf)
            if results and len(results) > 0 and results[0] is not None:
                any_detection = True
                s = format_result(results[0])
                pose_pubs[i].set(s)
                print(f"  cam{i}: {s}")
            else:
                pose_pubs[i].set("no detection")

        fps = fps_sub.get()
        if any_detection:
            print(f"  fps: {fps}")

        time.sleep(0.1)
