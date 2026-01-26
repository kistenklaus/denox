import time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

MODEL_PATH = "net.onnx"

BATCH = 1
C = 3
H = 3264
W = 4928

WARMUP = 20
ITER = 200

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(
    flags=1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
)
parser = trt.OnnxParser(network, logger)

with open(MODEL_PATH, "rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise SystemExit

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

if builder.platform_has_fast_fp16:
    config.set_flag(trt.BuilderFlag.FP16)

# ---------------------------------------------------
# Input/output names
# ---------------------------------------------------
input_name = network.get_input(0).name
output_name = network.get_output(0).name

# ---------------------------------------------------
# Optimization profile (required)
# ---------------------------------------------------
shape = (BATCH, C, H, W)
profile = builder.create_optimization_profile()
profile.set_shape(input_name, shape, shape, shape)
config.add_optimization_profile(profile)

# ---------------------------------------------------
# Build engine
# ---------------------------------------------------
serialized = builder.build_serialized_network(network, config)
if serialized is None:
    raise RuntimeError("Failed to build engine")

runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(serialized)
context = engine.create_execution_context()

context.set_input_shape(input_name, shape)
output_shape = context.get_tensor_shape(output_name)

# ---------------------------------------------------
# Allocate buffers
# ---------------------------------------------------
host_in = np.random.rand(*shape).astype(np.float32)
host_out = np.empty(output_shape, dtype=np.float32)

d_in = cuda.mem_alloc(host_in.nbytes)
d_out = cuda.mem_alloc(host_out.nbytes)

# **NEW API: assign device pointers**
context.set_tensor_address(input_name, int(d_in))
context.set_tensor_address(output_name, int(d_out))

stream = cuda.Stream()

# GPU-only timing events
start = cuda.Event()
end = cuda.Event()

# One-time transfer
cuda.memcpy_htod_async(d_in, host_in, stream)

# Warmup
for _ in range(WARMUP):
    context.execute_async_v3(stream.handle)
    stream.synchronize()

# Benchmark
times = []
for _ in range(ITER):
    start.record(stream)
    context.execute_async_v3(stream.handle)
    end.record(stream)
    end.synchronize()
    times.append(start.time_till(end))  # ms

# ---------------------------------------------------
# Stats
# ---------------------------------------------------
times_sorted = sorted(times)
mean = np.mean(times_sorted)
p50 = times_sorted[int(0.50 * ITER)]
p95 = times_sorted[int(0.95 * ITER)]
p99 = times_sorted[int(0.99 * ITER)]
fps = 1000.0 / mean

print(f"Mean latency: {mean:.3f} ms")
print(f"P50:          {p50:.3f} ms")
print(f"P95:          {p95:.3f} ms")
print(f"P99:          {p99:.3f} ms")
print(f"FPS:          {fps:.2f}")
