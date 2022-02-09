import onnxruntime as ort

ort_session = ort.InferenceSession("/home/bmw/anaconda3/envs/refinedetrepo2/refinedet.pytorch/refinedet320.onnx")

outputs = ort_session.run(
    None,
    {"input": np.random.randn(32, 3, 320, 320).astype(np.float32)},
)
print(outputs[0])