import ctypes
import ctypes.util
import math
import os
from typing import List, Optional, Tuple

import cv2
import numpy

from vision_types import ObjectDetectionObservation


class _CudaRuntime:
    cudaMemcpyHostToDevice = 1
    cudaMemcpyDeviceToHost = 2

    def __init__(self) -> None:
        library_name = ctypes.util.find_library("cudart") or "libcudart.so"
        self._lib = ctypes.CDLL(library_name)
        self._lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self._lib.cudaFree.argtypes = [ctypes.c_void_p]
        self._lib.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self._lib.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        self._lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        self._lib.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p,
        ]

    def _check(self, status: int, name: str) -> None:
        if status != 0:
            raise RuntimeError(f"{name} failed with CUDA status {status}")

    def malloc(self, size: int) -> ctypes.c_void_p:
        ptr = ctypes.c_void_p()
        self._check(self._lib.cudaMalloc(ctypes.byref(ptr), size), "cudaMalloc")
        return ptr

    def free(self, ptr: ctypes.c_void_p) -> None:
        if ptr:
            self._lib.cudaFree(ptr)

    def create_stream(self) -> ctypes.c_void_p:
        stream = ctypes.c_void_p()
        self._check(self._lib.cudaStreamCreate(ctypes.byref(stream)), "cudaStreamCreate")
        return stream

    def destroy_stream(self, stream: ctypes.c_void_p) -> None:
        if stream:
            self._lib.cudaStreamDestroy(stream)

    def memcpy_async(self, dst, src_ptr, size, direction, stream) -> None:
        self._lib.cudaMemcpyAsync(dst, ctypes.c_void_p(src_ptr), size, direction, stream)

    def synchronize(self, stream) -> None:
        self._check(self._lib.cudaStreamSynchronize(stream), "cudaStreamSynchronize")


class TensorRTYoloDetector:
    def __init__(
        self,
        model_path: str,
        camera_matrix: numpy.ndarray,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> None:
        self._camera_matrix = camera_matrix
        self._confidence_threshold = confidence_threshold
        self._iou_threshold = iou_threshold
        self._model_path = model_path
        self._initialized = False

    def _ensure_engine(self) -> str:
        """Return path to .engine file, exporting from .pt/.onnx if needed."""
        if self._model_path.endswith(".engine"):
            return self._model_path

        engine_path = self._model_path.rsplit(".", 1)[0] + ".engine"
        if os.path.exists(engine_path):
            return engine_path

        if self._model_path.endswith(".pt"):
            onnx_path = self._model_path.rsplit(".", 1)[0] + ".onnx"
            if not os.path.exists(onnx_path):
                print(f"Exporting {self._model_path} to ONNX...")
                os.environ["YOLO_AUTOINSTALL"] = "false"
                from ultralytics import YOLO
                YOLO(self._model_path).export(format="onnx")
            self._model_path = onnx_path

        if self._model_path.endswith(".onnx"):
            print(f"Building TensorRT engine from {self._model_path}...")
            self._build_engine(self._model_path, engine_path)

        return engine_path

    def _build_engine(self, onnx_path: str, engine_path: str) -> None:
        import tensorrt as trt
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                errors = [parser.get_error(i).desc() for i in range(parser.num_errors)]
                raise RuntimeError("Failed to parse ONNX: " + "; ".join(errors))

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        engine = builder.build_serialized_network(network, config)
        if engine is None:
            raise RuntimeError("TensorRT failed to build engine")

        with open(engine_path, "wb") as f:
            f.write(bytes(engine))
        print(f"Saved TensorRT engine to {engine_path}")

    def _initialize(self, frame_shape) -> None:
        if self._initialized:
            return

        import tensorrt as trt

        engine_path = self._ensure_engine()

        self._cuda = _CudaRuntime()
        self._stream = self._cuda.create_stream()

        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(engine_path, "rb") as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())

        self._context = self._engine.create_execution_context()

        # Find input tensor
        self._input_name = None
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self._input_name = name
                break

        input_shape = tuple(int(d) for d in self._engine.get_tensor_shape(self._input_name))
        if any(d < 0 for d in input_shape):
            h, w = frame_shape[0], frame_shape[1]
            runtime_shape = list(input_shape)
            runtime_shape[0] = 1
            if runtime_shape[2] < 0: runtime_shape[2] = h
            if runtime_shape[3] < 0: runtime_shape[3] = w
            input_shape = tuple(runtime_shape)
            self._context.set_input_shape(self._input_name, input_shape)

        self._input_shape = input_shape
        self._input_host, self._input_dev = self._alloc_binding(input_shape, trt.nptype(self._engine.get_tensor_dtype(self._input_name)))
        self._context.set_tensor_address(self._input_name, int(self._input_dev.value))

        # Output tensors
        self._outputs = []
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            if name == self._input_name:
                continue
            shape = tuple(int(d) for d in self._context.get_tensor_shape(name))
            dtype = trt.nptype(self._engine.get_tensor_dtype(name))
            host, dev = self._alloc_binding(shape, dtype)
            self._context.set_tensor_address(name, int(dev.value))
            self._outputs.append((name, shape, host, dev))

        self._initialized = True
        print(f"TensorRT engine loaded: input={self._input_shape}")

    def _alloc_binding(self, shape, dtype):
        host = numpy.empty(int(numpy.prod(shape)), dtype=dtype)
        dev = self._cuda.malloc(host.nbytes)
        return host, dev

    def detect(self, frame: cv2.Mat) -> List[ObjectDetectionObservation]:
        if not self._initialized:
            self._initialize(frame.shape)

        input_h, input_w = int(self._input_shape[2]), int(self._input_shape[3])
        blob, scale, pad_x, pad_y = self._preprocess(frame, input_w, input_h)

        numpy.copyto(self._input_host, blob.reshape(-1))
        self._cuda.memcpy_async(self._input_dev, self._input_host.ctypes.data,
                                self._input_host.nbytes, _CudaRuntime.cudaMemcpyHostToDevice, self._stream)

        self._context.execute_async_v3(int(self._stream.value))

        for _, shape, host, dev in self._outputs:
            self._cuda.memcpy_async(ctypes.c_void_p(host.ctypes.data), dev.value,
                                    host.nbytes, _CudaRuntime.cudaMemcpyDeviceToHost, self._stream)
        self._cuda.synchronize(self._stream)

        tensors = [host.reshape(shape).copy() for _, shape, host, _ in self._outputs]
        return self._postprocess(tensors, frame.shape[1], frame.shape[0], scale, pad_x, pad_y, input_w, input_h)

    @staticmethod
    def _preprocess(frame, input_w, input_h):
        color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if len(frame.shape) == 2 else frame
        fh, fw = color.shape[:2]
        scale = min(input_w / fw, input_h / fh)
        rw, rh = int(round(fw * scale)), int(round(fh * scale))
        resized = cv2.resize(color, (rw, rh), interpolation=cv2.INTER_LINEAR)
        canvas = numpy.full((input_h, input_w, 3), 114, dtype=numpy.uint8)
        px, py = (input_w - rw) // 2, (input_h - rh) // 2
        canvas[py:py+rh, px:px+rw] = resized
        blob = numpy.expand_dims(numpy.transpose(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), (2, 0, 1)).astype(numpy.float32) / 255.0, 0)
        return blob, scale, px, py

    def _postprocess(self, tensors, fw, fh, scale, px, py, iw, ih):
        if not tensors:
            return []

        tensor = max(tensors, key=lambda t: t.size)
        rows = numpy.squeeze(tensor)
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        elif rows.ndim > 2:
            rows = rows.reshape(rows.shape[-2], rows.shape[-1])
        if rows.shape[0] < rows.shape[1] and rows.shape[0] in (4, 5, 6, 7, 84, 85):
            rows = rows.transpose()
        rows = rows.astype(numpy.float32, copy=False)

        boxes, confs, cls_ids = [], [], []
        for row in rows:
            parsed = self._parse_row(row, iw, ih)
            if parsed is None:
                continue
            cx, cy, w, h, conf, cid = parsed
            if conf < self._confidence_threshold or w <= 0 or h <= 0:
                continue
            x0 = int(max(0, min(fw-1, round((cx - w/2 - px) / scale))))
            y0 = int(max(0, min(fh-1, round((cy - h/2 - py) / scale))))
            x1 = int(max(0, min(fw-1, round((cx + w/2 - px) / scale))))
            y1 = int(max(0, min(fh-1, round((cy + h/2 - py) / scale))))
            if x1 <= x0 or y1 <= y0:
                continue
            boxes.append([x0, y0, x1-x0, y1-y0])
            confs.append(float(conf))
            cls_ids.append(int(cid))

        if not boxes:
            return []

        kept = cv2.dnn.NMSBoxes(boxes, confs, self._confidence_threshold, self._iou_threshold)
        if len(kept) == 0:
            return []

        detections = []
        for idx in numpy.array(kept).reshape(-1):
            x0, y0, w, h = boxes[int(idx)]
            x1, y1 = x0 + w, y0 + h
            cx, cy = x0 + w/2.0, y0 + h/2.0
            pitch, yaw = self._pixel_to_angles(cx, cy)
            detections.append(ObjectDetectionObservation(
                class_id=cls_ids[int(idx)], confidence=confs[int(idx)],
                x0=x0, y0=y0, x1=x1, y1=y1,
                centroid_x=cx, centroid_y=cy, area_px=w*h,
                pitch_deg=pitch, yaw_deg=yaw,
            ))
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    @staticmethod
    def _parse_row(row, iw, ih):
        cols = row.shape[0]
        if cols < 6:
            return None
        if cols in (6, 7):
            x0, y0, x1, y1, conf = map(float, row[:5])
            cid = int(row[5])
            w, h = x1-x0, y1-y0
            if max(abs(x0), abs(x1), abs(y0), abs(y1)) <= 2.0:
                x0 *= iw; x1 *= iw; y0 *= ih; y1 *= ih; w = x1-x0; h = y1-y0
            return x0+w/2, y0+h/2, w, h, conf, cid
        xywh = row[:4].astype(float)
        if max(abs(xywh[0]), abs(xywh[1]), abs(xywh[2]), abs(xywh[3])) <= 2.0:
            xywh[0] *= iw; xywh[1] *= ih; xywh[2] *= iw; xywh[3] *= ih
        if cols == 84:
            scores = row[4:]; cid = int(numpy.argmax(scores)); conf = float(scores[cid])
        elif cols == 85:
            obj = float(row[4]); scores = row[5:]; cid = int(numpy.argmax(scores)); conf = obj * float(scores[cid])
        else:
            tail = row[5:]
            if row[4] <= 1.0 and tail.size > 0 and float(numpy.max(tail)) <= 1.0:
                obj = float(row[4]); scores = tail; cid = int(numpy.argmax(scores)); conf = obj * float(scores[cid])
            else:
                scores = row[4:]; cid = int(numpy.argmax(scores)); conf = float(scores[cid])
        return float(xywh[0]), float(xywh[1]), float(xywh[2]), float(xywh[3]), conf, cid

    def _pixel_to_angles(self, px, py):
        fx, fy = float(self._camera_matrix[0,0]), float(self._camera_matrix[1,1])
        cx, cy = float(self._camera_matrix[0,2]), float(self._camera_matrix[1,2])
        rx, ry = (px-cx)/fx, (py-cy)/fy
        return math.degrees(math.atan2(-ry, math.sqrt(1+rx*rx))), math.degrees(math.atan2(rx, 1.0))
