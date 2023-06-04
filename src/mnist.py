import matplotlib.pyplot as plt
import numpy as np
import onnx
import onnxruntime as ort
import pandas
import scipy

################################################################################
# Data Reading Code
################################################################################

TRAIN_DATA_FILE = "../MNIST_CSV/mnist_train.csv"
TEST_DATA_FILE = "../MNIST_CSV/mnist_test.csv"

COLUMN_NAMES = ["label"]
for i in range(28):
    for j in range(28):
        COLUMN_NAMES.append(f"pixel-{i}-{j}")

train_data = pandas.read_csv(TRAIN_DATA_FILE, header=None, names=COLUMN_NAMES)
test_data = pandas.read_csv(TEST_DATA_FILE, header=None, names=COLUMN_NAMES)

NUM_TRAIN_DATA_IMAGES = 60000
TRAIN_DATA_IMAGES = (
    train_data[COLUMN_NAMES[1:]]
    .to_numpy(dtype=np.float32)
    .reshape((60000, 1, 1, 28, 28))
)
TRAIN_DATA_LABELS = train_data["label"]

NUM_TEST_DATA_IMAGES = 10000
TEST_DATA_IMAGES = (
    test_data[COLUMN_NAMES[1:]]
    .to_numpy(dtype=np.float32)
    .reshape((10000, 1, 1, 28, 28))
)
TEST_DATA_LABELS = test_data["label"]

################################################################################
# Visualization Code
################################################################################

# for i in range(10000):
#     plt.imshow(TRAIN_DATA_IMAGES[i][0][0])
#     plt.show()

################################################################################
# Inference Code
################################################################################

MODEL = "../mnist/model.onnx"
session = ort.InferenceSession(MODEL)

predicted_values = np.zeros(
    NUM_TEST_DATA_IMAGES,
)
for i in range(NUM_TEST_DATA_IMAGES):
    result = session.run(["Plus214_Output_0"], {"Input3": TEST_DATA_IMAGES[i]})
    predicted_values[i] = np.argmax(result)

print(f"Accuracy is {np.mean(predicted_values == TEST_DATA_LABELS)}")

################################################################################
# ONNX Model Creation Code
################################################################################

# Load in the ONNX model protobuf
onnx_model = onnx.load(MODEL)

# # Print out the initializers in the graph
# for initializer in onnx_model.graph.initializer:
#     print(f"name: {initializer.name}")
#     print(f"data_type: {initializer.data_type} (i.e., {onnx.helper.tensor_dtype_to_np_dtype(initializer.data_type)})")
#     print(f"dims: {initializer.dims}")
#     print()

# Modify Nodes
for initializer in onnx_model.graph.initializer:
    if initializer.name == "Parameter193":
        # Dense layer weights (256 x 10)
        weights = np.array(initializer.float_data).reshape((256, 10))
        u, s, vh = scipy.sparse.linalg.svds(weights, k=8)
        initializer.float_data[:] = (u @ np.diag(s) @ vh).reshape((2560,))
    # if initializer.name == "Parameter87" or initializer.name == "Parameter5":
    #     weights = np.array(initializer.float_data).reshape(initializer.dims)
    #     feature_maps, channels, _, _ = weights.shape
    #     for feature_map in range(feature_maps):
    #         for channel in range(channels):
    #             # SVD on the single 5x5 kernel
    #             assert weights[feature_map][channel].shape == (5, 5)
    #             u, s, vh = scipy.sparse.linalg.svds(weights[feature_map][channel], k=4)
    #             print(np.sort(s))
    #             weights[feature_map][channel] = u @ np.diag(s) @ vh
    #     # initializer.float_data[:] = np.zeros((np.prod(initializer.dims),))
    #     initializer.float_data[:] = weights.reshape((np.prod(initializer.dims),))

MODEL2 = "convnets_modified.onnx"
onnx.save(onnx_model, MODEL2)

new_session = ort.InferenceSession(MODEL2)

for i in range(NUM_TEST_DATA_IMAGES):
    result = new_session.run(["Plus214_Output_0"], {"Input3": TEST_DATA_IMAGES[i]})
    # result = session.run(None, {"Input3": TEST_DATA_IMAGES[i]})
    predicted_values[i] = np.argmax(result)

print(f"New accuracy is {np.mean(predicted_values == TEST_DATA_LABELS)}")
