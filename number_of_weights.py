import onnx
import numpy as np

MODEL_PATH = "mnist/model.onnx"

if __name__ == "__main__":
    onnx_model = onnx.load(MODEL_PATH)

    number_of_weights = 0
    for initializer in onnx_model.graph.initializer:
        number_of_weights += np.prod(initializer.dims)
    
    print(f"Number of weights: {number_of_weights}")
