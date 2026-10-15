# C2

Vision coprocessor for a multi-camera V4L2 module. It captures every camera at
once, runs **pipelines** on the frames and/or on flatbuffer messages from the
robot, and publishes the results to NetworkTables as timestamped flatbuffers.

```
                  ┌────────────── NT: /<dev>/config/* (camera_id, exposure, ...)
                  ▼
MultiCameraCapture ──FrameSet──► PipelineManager ──► PipelineRunner (1 thread / pipeline)
      (capture/)                   (pipelines/)          │  ▲
                  │                                      │  └── NT: enabled, config/*, flatbuffer inputs
                  └──► DebugStreams (MJPEG)              └────► NT: output/* (flatbuffers), status/*
```

## Running

```sh
# once: build the AprilTag detector library
cmake -S src/pipelines/apriltag/aruco_nano -B src/pipelines/apriltag/aruco_nano/build
cmake --build src/pipelines/apriltag/aruco_nano/build
pip install -r requirements.txt

python src/main.py                # run from the repo root (config.json and calibrations live here)
```

`config.json` holds what C2 needs before NT is up:

| key | meaning |
| --- | --- |
| `device_id` | NT root table, e.g. `dsv0` → `/dsv0/...`, and the NT client name |
| `server_ip` | NT server (the robot) |
| `stream_port` | annotated MJPEG stream; raw camera *i* on `stream_port+1+i`; calibration upload on `stream_port+10` |
| `num_cameras` | cameras side by side in the module's frame |
| `camera_id`, `camera_resolution_width/height`, `camera_exposure`, `camera_gain` | capture defaults until the robot publishes `/<dev>/config/*`. Width is per camera |
| `pipelines_enabled` | default `enabled` per pipeline, e.g. `{"apriltag": true}` |
| `time_sync_probe_interval_s` | see [Timestamps](#timestamps). Default 10, 0 disables |

Per-camera intrinsics come from `calibration_<i>.yml` (upload them on the calibration page).

## Code map

All C2 source is in `src/`. `schema/` stays at the repo root because the robot
project builds its Java classes from `C2/schema/*.fbs` (its git submodule).

| path | what |
| --- | --- |
| `src/main.py` | wires everything together; the capture loop |
| `src/core/time_sync.py` | **all** timestamp logic (read its docstring) |
| `src/core/nt_io.py` | `FlatbufferPublisher` / `FlatbufferSubscriber`: the only way pipelines talk to NT |
| `src/core/nt_settings.py` | binds a settings dataclass to NT topics |
| `src/core/status.py` | heartbeat, capture fps, per-camera connected |
| `src/core/frames.py` | `FrameSet`: one capture, all cameras, one timestamp |
| `src/capture/` | V4L2 capture, exposure via `v4l2-ctl`, frozen-camera watchdog |
| `src/pipelines/pipeline.py` | `Pipeline` base class: what a pipeline is |
| `src/pipelines/runner.py` | runs pipelines; enable/disable/config/status over NT |
| `src/pipelines/apriltag/` | the AprilTag pipeline (detector + its aruco_nano C++ library, tag layout, solver, encoder) |
| `src/servers/` | MJPEG debug streams, calibration upload page |
| `schema/*.fbs` | flatbuffer schemas; Python is generated into `schema/<namespace>/` |
| `tests/` | `python -m unittest discover -s tests -t .` |

## NetworkTables layout

Everything lives under `/<device_id>` (`/dsv0` below).

| topic | type | who writes | meaning |
| --- | --- | --- | --- |
| `/dsv0/config/camera_id` | string | robot | V4L2 device (`"0"` = `/dev/video0`) |
| `/dsv0/config/camera_resolution_width` | int | robot | per-camera width |
| `/dsv0/config/camera_resolution_height` | int | robot | |
| `/dsv0/config/camera_exposure` | int | robot | |
| `/dsv0/config/camera_gain` | int | robot | |
| `/dsv0/pipelines/<name>/enabled` | bool | robot/dashboard | run this pipeline |
| `/dsv0/pipelines/<name>/config/<field>` | per field | robot/dashboard | one topic per settings field |
| `/dsv0/pipelines/<name>/output/<output>` | raw (flatbuffer) | C2 | results, NT-timestamped with capture time |
| `/dsv0/pipelines/<name>/status/active` | bool | C2 | enabled and running without errors |
| `/dsv0/pipelines/<name>/status/fps` | double | C2 | |
| `/dsv0/pipelines/<name>/status/latency_ms` | double | C2 | capture → results published |
| `/dsv0/pipelines/<name>/status/error` | string | C2 | last exception, `""` if none |
| `/dsv0/status/heartbeat` | int | C2 | +1 every 100 ms while C2 runs (own thread, ticks even if the camera stalls) |
| `/dsv0/status/capture_fps` | double | C2 | |
| `/dsv0/status/camera_<i>/connected` | bool | C2 | camera *i* is delivering images: a frame within 0.5 s and its image isn't perfectly uniform (a dead sensor outputs a flat image) |
| `/dsv0/status/time_sync/*` | int | C2 | `connection_offset_us`, `probe_offset_us`, `probe_rtt2_us`, `correction_us` |

C2 publishes defaults for `enabled` and `config/*` with `setDefault`, so they
appear in dashboards (AdvantageScope, Elastic, OutlineViewer) and can be edited
live. Anything the robot or a dashboard publishes always wins over C2's
default, even if C2 restarts afterwards.

### `apriltag` pipeline

* **Config:** `tag_layout` (string: `AprilTagFieldLayout` JSON) and `fiducial_size_m` (double).
* **Output:** `output/camera_<i>`, type string `dsv0_fb`, schema `schema/apriltag.fbs`.
  One message per camera per frame, including frames with no tags, so "nothing
  seen" can be told apart from "disconnected". It contains the field-to-camera
  pose (two candidates for single-tag observations) and the tag IDs used.

## Timestamps

**Short version: on the robot, use the NT value timestamp
(`TimestampedRaw.timestamp`, in µs). It is the frame's capture time on the
robot's own clock (FPGA time on a roboRIO, the same clock as
`Timer.getFPGATimestamp()` and `WPIUtilJNI.now()`). The flatbuffers carry no
timestamp of their own; the old `timestamp_us` fields are deprecated and
always read as 0.**

What C2 does, end to end:

1. **Capture.** Each frame is stamped with its V4L2 buffer timestamp
   (CLOCK_MONOTONIC, set by the driver when the frame arrived). That is
   ~30 ms earlier than when `read()` returns, which the old code missed. It
   is converted to the NT clock (`ntcore._now()`), which is *not*
   `time.time()`.
2. **Sync.** ntcore measures `server_time - local_time` once, when it
   connects, and adds it to every value timestamp it sends. It sends
   nothing until that offset exists. This was verified in ntcore's source
   (v2026.2.1) and with a two-process test.
3. **Drift.** ntcore never re-measures the offset during a connection, so
   clock drift between the coprocessor and roboRIO (tens of ppm) accumulates.
   For example, 30 ppm over a 20 minute connection is 36 ms. `TimeSync`
   briefly connects a second NT client every `time_sync_probe_interval_s`
   to get a fresh measurement, keeps the lowest-latency recent one, and
   pre-corrects each timestamp by the difference. After a reconnect (for
   example, a robot reboot that resets FPGA time), old measurements are
   discarded automatically. Each probe shows up on the robot as an NT client
   named `<device_id>-timesync` connecting and disconnecting; that's expected.
4. **Gating.** If C2 is not synced with the robot, results are dropped rather
   than sent with a wrong timestamp.
5. **NT inputs.** Messages pipelines receive from the robot have their
   timestamps converted back to C2's clock, using the same corrected offset.

`tests/test_time_sync.py` runs the coprocessor in a separate process whose
clock is shifted by 1000 s and runs 0.5% fast. It checks that every message
reaches the robot with a timestamp that's never in the future and is
typically within a few ms of delivery. It also shows that ntcore alone
drifts to ~30 ms off within 6 s at that rate.

To check the clocks on the real robot, watch `/dsv0/status/time_sync/correction_us`
over a match. Its slope is the real drift, and its value is the error that
would exist without the correction.

## Adding a pipeline

1. Write a flatbuffer schema in `schema/` and generate code: `flatc --python -o schema schema/<file>.fbs`.
2. Subclass `Pipeline` (see the docstring in `src/pipelines/pipeline.py` for a full example):
   * `NAME`: its NT table.
   * `SETTINGS`: a frozen dataclass; each field becomes `config/<field>`.
   * `USES_CAMERAS`: whether to run on every frame set, or only when
     `NT_INPUTS` (flatbuffer topics from the robot or other pipelines) have new messages.
   * `process(inputs)`: publish with `context.output(name, type_string).publish(payload, context.time_sync.stamp(capture_time))`.
3. Add it to `PIPELINE_TYPES` in `src/pipelines/__init__.py`.

Each pipeline gets its own thread and is only constructed while enabled,
so a disabled pipeline costs nothing. If a pipeline falls behind, it skips
to the newest frame.
