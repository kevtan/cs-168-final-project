import numpy as np
import onnx

if __name__ == "__main__":
    MODEL_PATH = "googlenet-12/googlenet-12.onnx"

    onnx_model = onnx.load(MODEL_PATH)

    # Maps an initializer name to its number of weights
    total_number_of_weights = 0
    total_weights_in_2d_initializers = 0
    initializer_sizes = {}
    for initializer in onnx_model.graph.initializer:
        if len(initializer.dims) == 2:
            total_weights_in_2d_initializers += np.prod(initializer.dims)
        total_number_of_weights += np.prod(initializer.dims)
        initializer_sizes[initializer.name] = np.prod(initializer.dims)
    print(
        "Percentage of weights in 2D initializers: ",
        total_weights_in_2d_initializers / total_number_of_weights,
    )

    # Print initializers in order of size
    weights_seen_so_far = 0
    for initializer_name, initializer_size in sorted(
        initializer_sizes.items(), key=lambda x: x[1], reverse=True
    ):
        weights_seen_so_far += initializer_size
        print(
            f"({weights_seen_so_far / total_number_of_weights})\t{initializer_name}: {initializer_size}"
        )
