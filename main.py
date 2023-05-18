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
#     plt.imshow(raw_train_data[i][0][0])
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
    predicted_values[i] = np.argmax(result)

print(f"Accuracy is {np.mean(predicted_values == TEST_DATA_LABELS)}")

