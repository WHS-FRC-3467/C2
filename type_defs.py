import numpy
import numpy.typing

FloatArray = numpy.typing.NDArray[numpy.float64]
ImageArray = numpy.typing.NDArray[numpy.uint8]
IntArray = numpy.typing.NDArray[numpy.int32]


def empty_float_array() -> FloatArray:
    return numpy.empty((0,), dtype=numpy.float64)


def empty_image() -> ImageArray:
    return numpy.empty((0, 0), dtype=numpy.uint8)
