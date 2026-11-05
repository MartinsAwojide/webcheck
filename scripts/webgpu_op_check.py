"""Every operator type of the Studio graphs on ONNX Runtime's WebGPU provider, for CI on Linux
through Mesa's software Vulkan driver (lavapipe) and for any machine with the `webgpu` extra. One small graph (opset 12,
IR 8, as the exports) holds those operators in the shapes they are used in; it runs on the CPU and on WebGPU with CPU
fallback switched off, so the check fails if any operator does not run on WebGPU or if the outputs differ.

    uv run --with onnx python scripts/webgpu_op_check.py          (exit 0: every operator ran on WebGPU, outputs agree)
"""
import sys

import numpy as np
import onnx
import onnxruntime as ort
import onnxruntime_ep_webgpu as webgpu_ep
from onnx import TensorProto, helper, numpy_helper

TOLERANCE = 1e-4


def build_graph() -> bytes:
    rng = np.random.default_rng(0)
    c = lambda name, arr: numpy_helper.from_array(np.asarray(arr), name)
    inits = [c("pads", np.array([0, 0, 1, 1, 0, 0, 1, 1], np.int64)), c("W1", rng.normal(0, .2, (8, 3, 3, 3)).astype(np.float32)),
             c("B1", rng.normal(0, .1, 8).astype(np.float32)), c("W2", rng.normal(0, .2, (8, 8, 3, 3)).astype(np.float32)),
             c("lo", np.array(0, np.float32)), c("hi", np.array(6, np.float32)), c("six", np.array(6, np.float32)),
             c("roi", np.array([], np.float32)), c("scales", np.array([1, 1, 2, 2], np.float32)),
             c("shape2", np.array([128, 8], np.int64)), c("bank", rng.normal(0, 1, (8, 20)).astype(np.float32)),
             c("k", np.array([3], np.int64)), c("idx1", np.array([1], np.int64)), c("one", np.array(1, np.float32)),
             c("st", np.array([0], np.int64)), c("en", np.array([4], np.int64)), c("ax3", np.array([3], np.int64))]
    n = helper.make_node
    nodes = [  # the backbone's and YOLO-Pro's: a Conv without pads, a Pad, Clip, a Min against a constant, swish, pools
        n("Transpose", ["x"], ["t"], perm=[0, 3, 1, 2]), n("Pad", ["t", "pads"], ["p"]),
        n("Conv", ["p", "W1", "B1"], ["c1"], kernel_shape=[3, 3]), n("Clip", ["c1", "lo", "hi"], ["r"]),
        n("Conv", ["r", "W2"], ["c2"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]), n("Min", ["c2", "six"], ["m"]),
        n("Sigmoid", ["m"], ["s"]), n("Mul", ["m", "s"], ["h"]), n("Add", ["h", "r"], ["a"]),
        n("AveragePool", ["a"], ["ap"], kernel_shape=[2, 2], strides=[2, 2]),
        n("Resize", ["ap", "roi", "scales"], ["rs"], mode="nearest"),
        n("MaxPool", ["rs"], ["mp"], kernel_shape=[2, 2], strides=[2, 2]),
        n("Concat", ["ap", "mp"], ["cat"], axis=1), n("Split", ["cat"], ["sp1", "sp2"], axis=1, split=[8, 8]),
        # YOLO-Pro's two Einsum equations and its Softmax
        n("Einsum", ["sp1", "sp2"], ["e1"], equation="abcd,aecd->abce"),
        n("Einsum", ["e1", "e1"], ["e2"], equation="abcd,aecb->aecd"), n("Softmax", ["e2"], ["sm"], axis=-1),
        # the gate head: distances to a memory bank, the nearest ones
        n("Reshape", ["sm", "shape2"], ["f"]), n("MatMul", ["f", "bank"], ["mm"]), n("Neg", ["mm"], ["neg"]),
        n("TopK", ["neg", "k"], ["tv", "ti"], axis=-1), n("Neg", ["tv"], ["ng"]),
        n("ReduceSum", ["ng"], ["rsum"], axes=[1], keepdims=0), n("Relu", ["rsum"], ["rl"]), n("Sqrt", ["rl"], ["sq"]),
        n("Sub", ["sq", "one"], ["out1"]),
        # shape handling
        n("Shape", ["sm"], ["shp"]), n("Gather", ["shp", "idx1"], ["g"]), n("Cast", ["g"], ["out2"], to=TensorProto.FLOAT),
        n("Slice", ["sm", "st", "en", "ax3"], ["sl"]), n("Squeeze", ["sl"], ["out3"], axes=[0])]
    outputs = [helper.make_tensor_value_info(o, TensorProto.FLOAT, s) for o, s in (("out1", [128]), ("out2", [1]),
                                                                                   ("out3", [8, 16, 4]))]
    graph = helper.make_graph(nodes, "webcheck_ops", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 32, 32, 3])],
                              outputs, inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)], ir_version=8)
    onnx.checker.check_model(model)
    return model.SerializeToString()


def main() -> int:
    blob = build_graph()
    ops = sorted({node.op_type for node in onnx.load_from_string(blob).graph.node})
    x = np.random.default_rng(1).random((1, 32, 32, 3), dtype=np.float32)
    cpu = ort.InferenceSession(blob, providers=["CPUExecutionProvider"]).run(None, {"x": x})
    ort.register_execution_provider_library("webgpu", webgpu_ep.get_library_path())
    devices = [d for d in ort.get_ep_devices() if d.ep_name == webgpu_ep.get_ep_name()]
    print(f"onnxruntime {ort.__version__}; {len(ops)} operator types: {', '.join(ops)}")
    if not devices:
        print("FAIL: no WebGPU device (on Linux, is a Vulkan driver installed?)")
        return 1
    options = ort.SessionOptions()
    options.add_provider_for_devices(devices[:1], {})
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")      # every operator on WebGPU, or fail
    try:
        gpu = ort.InferenceSession(blob, sess_options=options).run(None, {"x": x})
    except Exception as exc:
        print(f"FAIL: not every operator runs on WebGPU: {str(exc).splitlines()[0]}")
        return 1
    diff = max(float(np.max(np.abs(a - b))) for a, b in zip(cpu, gpu))
    ok = diff <= TOLERANCE
    print(f"{'PASS' if ok else 'FAIL'}: every operator ran on WebGPU; outputs within {diff:.2e} of the CPU "
          f"(tolerance {TOLERANCE:g})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
