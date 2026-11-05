"""Board-side inference: frame tiling, preprocessing, the FOMO-AD gate score, YOLO-Pro decoding and the
TensorRT engine cache. Both models run on the board's host TensorRT from Edge Impulse's exported ONNX.
Everything except `trt` is pure numpy and runs under pytest without a GPU."""
