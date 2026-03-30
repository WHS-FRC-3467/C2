import ctypes
import ctypes.util
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy

from vision_types import ObjectDetectionObservation


class TensorRTDetectorError(RuntimeError):
    pass


@dataclass
class _TensorBinding:
    name: str
    shape: Tuple[int, ...]
    dtype: numpy.dtype
    size_bytes: int
    host: numpy.ndarray
    device_ptr: ctypes.c_void_p


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
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]

    def _check(self, status: int, call_name: str) -> None:
        if status != 0:
            raise TensorRTDetectorError(f"{call_name} failed with CUDA status {status}")

    def malloc(self, size_bytes: int) -> ctypes.c_void_p:
        ptr = ctypes.c_void_p()
        self._check(self._lib.cudaMalloc(ctypes.byref(ptr), size_bytes), "cudaMalloc")
        return ptr

    def free(self, ptr: ctypes.c_void_p) -> None:
        if ptr:
            self._check(self._lib.cudaFree(ptr), "cudaFree")

    def create_stream(self) -> ctypes.c_void_p:
        stream = ctypes.c_void_p()
        self._check(self._lib.cudaStreamCreate(ctypes.byref(stream)), "cudaStreamCreate")
        return stream

    def destroy_stream(self, stream: ctypes.c_void_p) -> None:
        if stream:
            self._check(self._lib.cudaStreamDestroy(stream), "cudaStreamDestroy")

    def memcpy_async(self, dst: ctypes.c_void_p, src_ptr: int, size_bytes: int, direction: int, stream: ctypes.c_void_p) -> None:
        self._check(
            self._lib.cudaMemcpyAsync(dst, ctypes.c_void_p(src_ptr), size_bytes, direction, stream),
            "cudaMemcpyAsync",
        )

    def synchronize(self, stream: ctypes.c_void_p) -> None:
        self._check(self._lib.cudaStreamSynchronize(stream), "cudaStreamSynchronize")


class TensorRTYoloDetector:
    def __init__(
        self,
        engine_path: str,
        onnx_path: str,
        camera_matrix: numpy.ndarray,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        workspace_bytes: int = 1 << 30,
    ) -> None:
        self._engine_path = engine_path
        self._onnx_path = onnx_path
        self._camera_matrix = camera_matrix
        self._confidence_threshold = confidence_threshold
        self._iou_threshold = iou_threshold
        self._workspace_bytes = workspace_bytes
        self._trt = None
        self._logger = None
        self._cuda: Optional[_CudaRuntime] = None
        self._runtime = None
        self._engine = None
        self._context = None
        self._stream: Optional[ctypes.c_void_p] = None
        self._input_name: Optional[str] = None
        self._input_shape: Optional[Tuple[int, ...]] = None
        self._input_binding: Optional[_TensorBinding] = None
        self._output_bindings: List[_TensorBinding] = []
        self._initialized = False

    def initialize(self, frame_shape: Sequence[int]) -> None:
        if self._initialized:
            return

        if self._camera_matrix.size == 0:
            raise TensorRTDetectorError("Camera matrix for video1 is not available")

        try:
            import tensorrt as trt  # type: ignore
        except ModuleNotFoundError as exc:
            raise TensorRTDetectorError("TensorRT Python bindings are required to run the YOLO detector") from exc

        self._trt = trt
        self._logger = trt.Logger(trt.Logger.WARNING)
        self._cuda = _CudaRuntime()
        self._stream = self._cuda.create_stream()

        frame_height, frame_width = int(frame_shape[0]), int(frame_shape[1])
        if not os.path.exists(self._engine_path):
            self._build_engine(frame_height, frame_width)

        self._load_engine(frame_height, frame_width)
        self._initialized = True

    def release(self) -> None:
        if self._cuda is not None:
            for binding in self._output_bindings:
                self._cuda.free(binding.device_ptr)
            self._output_bindings = []
            if self._input_binding is not None:
                self._cuda.free(self._input_binding.device_ptr)
                self._input_binding = None
            if self._stream is not None:
                self._cuda.destroy_stream(self._stream)
                self._stream = None

        self._context = None
        self._engine = None
        self._runtime = None
        self._initialized = False

    def detect(self, frame: cv2.Mat) -> List[ObjectDetectionObservation]:
        if not self._initialized:
            self.initialize(frame.shape)

        if self._context is None or self._input_binding is None or self._cuda is None or self._stream is None:
            raise TensorRTDetectorError("Detector is not initialized")

        input_height, input_width = self._network_input_hw()
        blob, scale, pad_x, pad_y = self._preprocess(frame, input_width, input_height)

        numpy.copyto(self._input_binding.host, blob.reshape(-1))
        self._cuda.memcpy_async(
            self._input_binding.device_ptr,
            self._input_binding.host.ctypes.data,
            self._input_binding.size_bytes,
            _CudaRuntime.cudaMemcpyHostToDevice,
            self._stream,
        )

        if not self._context.execute_async_v3(int(self._stream.value)):
            raise TensorRTDetectorError("TensorRT execute_async_v3 failed")

        for binding in self._output_bindings:
            self._cuda.memcpy_async(
                ctypes.c_void_p(binding.host.ctypes.data),
                binding.device_ptr.value,
                binding.size_bytes,
                _CudaRuntime.cudaMemcpyDeviceToHost,
                self._stream,
            )
        self._cuda.synchronize(self._stream)

        output_tensors = [binding.host.reshape(binding.shape).copy() for binding in self._output_bindings]
        return self._postprocess(output_tensors, frame.shape[1], frame.shape[0], scale, pad_x, pad_y)

    def _build_engine(self, frame_height: int, frame_width: int) -> None:
        if not os.path.exists(self._onnx_path):
            raise TensorRTDetectorError(
                f"Missing engine at {self._engine_path} and ONNX source at {self._onnx_path}"
            )

        trt = self._trt
        builder = trt.Builder(self._logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, self._logger)
        with open(self._onnx_path, "rb") as onnx_file:
            if not parser.parse(onnx_file.read()):
                errors = [parser.get_error(i).desc() for i in range(parser.num_errors)]
                raise TensorRTDetectorError("Failed to parse ONNX: " + "; ".join(errors))

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, self._workspace_bytes)
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        if network.num_inputs != 1:
            raise TensorRTDetectorError(f"Expected one input tensor, found {network.num_inputs}")

        input_tensor = network.get_input(0)
        input_shape = tuple(int(dim) for dim in input_tensor.shape)
        if any(dim < 0 for dim in input_shape):
            profile = builder.create_optimization_profile()
            target_shape = self._shape_for_frame(input_shape, frame_height, frame_width)
            profile.set_shape(input_tensor.name, target_shape, target_shape, target_shape)
            config.add_optimization_profile(profile)

        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise TensorRTDetectorError("TensorRT failed to build a serialized engine")

        with open(self._engine_path, "wb") as engine_file:
            engine_file.write(bytes(serialized_engine))

    def _load_engine(self, frame_height: int, frame_width: int) -> None:
        trt = self._trt
        self._runtime = trt.Runtime(self._logger)
        with open(self._engine_path, "rb") as engine_file:
            engine_data = engine_file.read()
        self._engine = self._runtime.deserialize_cuda_engine(engine_data)
        if self._engine is None:
            raise TensorRTDetectorError(f"Failed to deserialize TensorRT engine {self._engine_path}")

        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise TensorRTDetectorError("Failed to create TensorRT execution context")

        input_names = [
            self._engine.get_tensor_name(i)
            for i in range(self._engine.num_io_tensors)
            if self._engine.get_tensor_mode(self._engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT
        ]
        if len(input_names) != 1:
            raise TensorRTDetectorError(f"Expected one engine input tensor, found {len(input_names)}")
        self._input_name = input_names[0]

        input_shape = tuple(int(dim) for dim in self._engine.get_tensor_shape(self._input_name))
        if any(dim < 0 for dim in input_shape):
            runtime_shape = self._shape_for_frame(input_shape, frame_height, frame_width)
            if not self._context.set_input_shape(self._input_name, runtime_shape):
                raise TensorRTDetectorError(f"Failed to set runtime input shape for {self._input_name}")
            input_shape = tuple(int(dim) for dim in self._context.get_tensor_shape(self._input_name))
        self._input_shape = input_shape

        self._input_binding = self._allocate_binding(self._input_name, input_shape)
        self._context.set_tensor_address(self._input_name, int(self._input_binding.device_ptr.value))

        self._output_bindings = []
        for tensor_index in range(self._engine.num_io_tensors):
            tensor_name = self._engine.get_tensor_name(tensor_index)
            if tensor_name == self._input_name:
                continue
            tensor_shape = tuple(int(dim) for dim in self._context.get_tensor_shape(tensor_name))
            if any(dim < 0 for dim in tensor_shape):
                raise TensorRTDetectorError(f"Dynamic output shape for {tensor_name} is not resolved")
            binding = self._allocate_binding(tensor_name, tensor_shape)
            self._context.set_tensor_address(tensor_name, int(binding.device_ptr.value))
            self._output_bindings.append(binding)

    def _allocate_binding(self, tensor_name: str, tensor_shape: Tuple[int, ...]) -> _TensorBinding:
        dtype = numpy.dtype(self._trt.nptype(self._engine.get_tensor_dtype(tensor_name)))
        host = numpy.empty(int(numpy.prod(tensor_shape)), dtype=dtype)
        size_bytes = int(host.nbytes)
        device_ptr = self._cuda.malloc(size_bytes)
        return _TensorBinding(tensor_name, tensor_shape, dtype, size_bytes, host, device_ptr)

    @staticmethod
    def _shape_for_frame(input_shape: Tuple[int, ...], frame_height: int, frame_width: int) -> Tuple[int, ...]:
        target_shape = list(input_shape)
        if len(target_shape) == 4:
            target_shape[0] = 1 if target_shape[0] < 0 else target_shape[0]
            if target_shape[2] < 0:
                target_shape[2] = frame_height
            if target_shape[3] < 0:
                target_shape[3] = frame_width
        elif len(target_shape) == 3:
            if target_shape[1] < 0:
                target_shape[1] = frame_height
            if target_shape[2] < 0:
                target_shape[2] = frame_width
        else:
            raise TensorRTDetectorError(f"Unsupported input rank: {len(target_shape)}")
        return tuple(int(dim) for dim in target_shape)

    def _network_input_hw(self) -> Tuple[int, int]:
        if self._input_shape is None:
            raise TensorRTDetectorError("Input shape is not available")
        if len(self._input_shape) == 4:
            return int(self._input_shape[2]), int(self._input_shape[3])
        if len(self._input_shape) == 3:
            return int(self._input_shape[1]), int(self._input_shape[2])
        raise TensorRTDetectorError(f"Unsupported input rank: {len(self._input_shape)}")

    @staticmethod
    def _preprocess(frame: cv2.Mat, input_width: int, input_height: int) -> Tuple[numpy.ndarray, float, int, int]:
        if len(frame.shape) == 2:
            color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            color = frame

        frame_height, frame_width = color.shape[:2]
        scale = min(input_width / frame_width, input_height / frame_height)
        resized_width = int(round(frame_width * scale))
        resized_height = int(round(frame_height * scale))
        resized = cv2.resize(color, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        canvas = numpy.full((input_height, input_width, 3), 114, dtype=numpy.uint8)
        pad_x = (input_width - resized_width) // 2
        pad_y = (input_height - resized_height) // 2
        canvas[pad_y:pad_y + resized_height, pad_x:pad_x + resized_width] = resized

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        chw = numpy.transpose(rgb, (2, 0, 1)).astype(numpy.float32) / 255.0
        return numpy.expand_dims(chw, axis=0), scale, pad_x, pad_y

    def _postprocess(
        self,
        output_tensors: List[numpy.ndarray],
        frame_width: int,
        frame_height: int,
        scale: float,
        pad_x: int,
        pad_y: int,
    ) -> List[ObjectDetectionObservation]:
        if not output_tensors:
            return []

        tensor = max(output_tensors, key=lambda item: item.size)
        candidates = self._normalize_output_rows(tensor)
        boxes_xywh: List[List[int]] = []
        confidences: List[float] = []
        class_ids: List[int] = []

        input_height, input_width = self._network_input_hw()

        for row in candidates:
            parsed = self._parse_candidate_row(row, input_width, input_height)
            if parsed is None:
                continue
            x_center, y_center, width, height, confidence, class_id = parsed
            if confidence < self._confidence_threshold or width <= 0 or height <= 0:
                continue

            x0 = (x_center - width / 2 - pad_x) / scale
            y0 = (y_center - height / 2 - pad_y) / scale
            x1 = (x_center + width / 2 - pad_x) / scale
            y1 = (y_center + height / 2 - pad_y) / scale

            x0 = int(max(0, min(frame_width - 1, round(x0))))
            y0 = int(max(0, min(frame_height - 1, round(y0))))
            x1 = int(max(0, min(frame_width - 1, round(x1))))
            y1 = int(max(0, min(frame_height - 1, round(y1))))
            if x1 <= x0 or y1 <= y0:
                continue

            boxes_xywh.append([x0, y0, x1 - x0, y1 - y0])
            confidences.append(float(confidence))
            class_ids.append(int(class_id))

        if not boxes_xywh:
            return []

        kept = cv2.dnn.NMSBoxes(boxes_xywh, confidences, self._confidence_threshold, self._iou_threshold)
        if len(kept) == 0:
            return []

        detections: List[ObjectDetectionObservation] = []
        for index in numpy.array(kept).reshape(-1):
            x0, y0, width, height = boxes_xywh[int(index)]
            x1 = x0 + width
            y1 = y0 + height
            centroid_x = x0 + width / 2.0
            centroid_y = y0 + height / 2.0
            pitch_deg, yaw_deg = self._pixel_to_angles(centroid_x, centroid_y)
            detections.append(
                ObjectDetectionObservation(
                    class_id=class_ids[int(index)],
                    confidence=confidences[int(index)],
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    centroid_x=centroid_x,
                    centroid_y=centroid_y,
                    area_px=int(width * height),
                    pitch_deg=pitch_deg,
                    yaw_deg=yaw_deg,
                )
            )

        detections.sort(key=lambda detection: detection.confidence, reverse=True)
        return detections

    @staticmethod
    def _normalize_output_rows(tensor: numpy.ndarray) -> numpy.ndarray:
        rows = numpy.squeeze(tensor)
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        elif rows.ndim > 2:
            rows = rows.reshape(rows.shape[-2], rows.shape[-1])

        if rows.ndim != 2:
            raise TensorRTDetectorError(f"Unsupported YOLO output rank: {rows.ndim}")

        if rows.shape[0] < rows.shape[1] and rows.shape[0] in (4, 5, 6, 7, 84, 85):
            rows = rows.transpose()
        return rows.astype(numpy.float32, copy=False)

    def _parse_candidate_row(
        self,
        row: numpy.ndarray,
        input_width: int,
        input_height: int,
    ) -> Optional[Tuple[float, float, float, float, float, int]]:
        cols = row.shape[0]
        if cols < 6:
            return None

        if cols in (6, 7):
            x0, y0, x1, y1, confidence = map(float, row[:5])
            class_id = int(row[5])
            width = x1 - x0
            height = y1 - y0
            if max(abs(x0), abs(x1), abs(y0), abs(y1)) <= 2.0:
                x0 *= input_width
                x1 *= input_width
                y0 *= input_height
                y1 *= input_height
                width = x1 - x0
                height = y1 - y0
            return x0 + width / 2.0, y0 + height / 2.0, width, height, confidence, class_id

        xywh = row[:4].astype(float)
        if max(abs(xywh[0]), abs(xywh[1]), abs(xywh[2]), abs(xywh[3])) <= 2.0:
            xywh[0] *= input_width
            xywh[1] *= input_height
            xywh[2] *= input_width
            xywh[3] *= input_height

        if cols == 84:
            class_scores = row[4:]
            class_id = int(numpy.argmax(class_scores))
            confidence = float(class_scores[class_id])
        elif cols == 85:
            objectness = float(row[4])
            class_scores = row[5:]
            class_id = int(numpy.argmax(class_scores))
            confidence = objectness * float(class_scores[class_id])
        else:
            tail = row[5:]
            if row[4] <= 1.0 and tail.size > 0 and float(numpy.max(tail)) <= 1.0:
                objectness = float(row[4])
                class_scores = tail
                class_id = int(numpy.argmax(class_scores))
                confidence = objectness * float(class_scores[class_id])
            else:
                class_scores = row[4:]
                class_id = int(numpy.argmax(class_scores))
                confidence = float(class_scores[class_id])

        return float(xywh[0]), float(xywh[1]), float(xywh[2]), float(xywh[3]), confidence, class_id

    def _pixel_to_angles(self, pixel_x: float, pixel_y: float) -> Tuple[float, float]:
        fx = float(self._camera_matrix[0, 0])
        fy = float(self._camera_matrix[1, 1])
        cx = float(self._camera_matrix[0, 2])
        cy = float(self._camera_matrix[1, 2])

        ray_x = (pixel_x - cx) / fx
        ray_y = (pixel_y - cy) / fy

        yaw_deg = math.degrees(math.atan2(ray_x, 1.0))
        pitch_deg = math.degrees(math.atan2(-ray_y, math.sqrt(1.0 + ray_x * ray_x)))
        return pitch_deg, yaw_deg
