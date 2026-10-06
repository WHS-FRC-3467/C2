import tensorrt as trt

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(
    1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
)
parser = trt.OnnxParser(network, logger)

with open("model.onnx", "rb") as f:
    parser.parse(f.read())

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
if builder.platform_has_fast_fp16:
    config.set_flag(trt.BuilderFlag.FP16)

engine = builder.build_serialized_network(network, config)
with open("model.engine", "wb") as f:
    f.write(bytes(engine))

print("Done: model.engine")
