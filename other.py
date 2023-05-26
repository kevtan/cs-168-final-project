import matplotlib.pyplot as plt
import numpy as np
import onnx
import onnxruntime as ort
import pandas

################################################################################
# Data Reading Code
################################################################################

TRAIN_DATA_FILE = "MNIST_CSV/mnist_train.csv"
TEST_DATA_FILE = "MNIST_CSV/mnist_test.csv"

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

MODEL = "mnist/model.onnx"
session = ort.InferenceSession(MODEL)

predicted_values = np.zeros(
    NUM_TEST_DATA_IMAGES,
)
for i in range(NUM_TEST_DATA_IMAGES):
    result = session.run(["Plus214_Output_0"], {"Input3": TEST_DATA_IMAGES[i]})
    # result = session.run(None, {"Input3": TEST_DATA_IMAGES[i]})
    predicted_values[i] = np.argmax(result)

print(f"Accuracy is {np.mean(predicted_values == TEST_DATA_LABELS)}")

################################################################################
# ONNX Model Creation Code
################################################################################

# Load in the ONNX model protobuf
onnx_model = onnx.load(MODEL)
# breakpoint()

weights = onnx.load(MODEL).graph.initializer
# "Parameter87" = weights[0]
# breakpoint()

# Modify Nodes
# Load the ONNX model
model = onnx.load(MODEL)
graph = model.graph
nodes = graph.node

for weight in weights:
  if weight.name == "Parameter87":
    # print(weight.float_data)
    breakpoint()
    print("before: ")
    print(weight.dims)
    weight.float_data = 


# Save the modified ONNX model
# onnx.save(model, "modified_model.onnx")