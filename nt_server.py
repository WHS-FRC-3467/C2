"""Standalone NetworkTables server that publishes config for both aruco and object detection."""

import json
import math
import sys
import os
import time

import ntcore

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "schema"))

from dsv0.Frame import Frame
from dsv0.Vec3 import Vec3
from dsv0.Quaternion import Quaternion

ARUCO_DEVICE_ID = "dsv0"
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


def parse_frame(buf):
    if not buf or len(buf) == 0:
        return None

    frame = Frame.GetRootAsFrame(buf, 0)
    results = []
    for i in range(frame.CamerasLength()):
        cam = frame.Cameras(i)

        assert cam is not None
        cam_obs = cam.CameraObservation()
        if cam_obs is None:
            results.append(None)
            continue

        sol = cam_obs.Solution0()
        if sol is None:
            results.append(None)
            continue

        v = Vec3()

        sol_pose = sol.Pose()
        assert sol_pose is not None
        sol_pose.Translation(v)
        q = Quaternion()
        sol_pose.Rotation(q)

        tag_ids = [cam_obs.TagIds(j) for j in range(cam_obs.TagIdsLength())]

        entry = {
            "camera_index": cam.CameraIndex(),
            "translation": {"x": v.X(), "y": v.Y(), "z": v.Z()},
            "rotation": {"qw": q.W(), "qx": q.X(), "qy": q.Y(), "qz": q.Z()},
            "error": sol.Error(),
            "tag_ids": tag_ids,
            "fps": cam.Fps(),
        }

        sol1 = cam_obs.Solution1()
        if sol1 is not None:
            entry["alt_error"] = sol1.Error()

        results.append(entry)
    return results


def format_result(r):
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


def format_detection(d):
    return (
        f"cls={d['class_id']} conf={d['confidence']:.2f} "
        f"[{d['x0']},{d['y0']}-{d['x1']},{d['y1']}] "
        f"yaw={d['yaw']:+.1f}° pitch={d['pitch']:+.1f}° area={d['area']}"
    )


def setup_aruco(inst):
    table = inst.getTable(f"/{ARUCO_DEVICE_ID}/config")
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

    obs_subs = []
    for i in range(NUM_CAMERAS):
        cam_table = inst.getTable(f"/{ARUCO_DEVICE_ID}/output/camera_{i}")
        obs_subs.append(
            cam_table.getRawTopic("observation").subscribe("dsv0_fb", bytes())
        )

    pose_table = inst.getTable(f"/{ARUCO_DEVICE_ID}/poses")
    pose_pubs = [
        pose_table.getStringTopic(f"camera_{i}").publish() for i in range(NUM_CAMERAS)
    ]
    fps_sub = (
        inst.getTable(f"/{ARUCO_DEVICE_ID}/output/camera_0")
        .getIntegerTopic("fps")
        .subscribe(0)
    )

    print(f"Published aruco config to /{ARUCO_DEVICE_ID}/config")

    return publish_config, obs_subs, pose_pubs, fps_sub


if __name__ == "__main__":
    inst = ntcore.NetworkTableInstance.getDefault()
    inst.startServer()
    print("NetworkTables server started (aruco + object detection)")

    aruco_publish, obs_subs, pose_pubs, fps_sub = setup_aruco(inst)

    print("Press Ctrl+C to stop\n")

    last_connections = 0
    while True:
        connections = inst.getConnections()
        if len(connections) != last_connections:
            last_connections = len(connections)
            print(f"Connections: {last_connections}")
            for c in connections:
                print(f"  {c.remote_id} @ {c.remote_ip}")
            aruco_publish()

        # Aruco results
        any_aruco = False
        for i in range(NUM_CAMERAS):
            buf = obs_subs[i].get()
            if not buf or len(buf) == 0:
                pose_pubs[i].set("no detection")
                continue

            results = parse_frame(buf)
            if results and len(results) > 0 and results[0] is not None:
                any_aruco = True
                s = format_result(results[0])
                pose_pubs[i].set(s)
                print(f"  aruco cam{i}: {s}")
            else:
                pose_pubs[i].set("no detection")

        if any_aruco:
            print(f"  aruco fps: {fps_sub.get()}")

        time.sleep(0.1)
