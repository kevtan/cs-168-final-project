import os

import matplotlib.pyplot as plt
import numpy as np
import onnx
import onnxruntime as ort
import pandas
import scipy
import seaborn as sns

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

# Reduced ranks for the `fc.weight` initializer.
K_VALUES = range(1, 10, 1)

acc_list = []

"""
for k in K_VALUES:
    MODIFIED_MODEL = f"../mnist/mnist-modified-{k}.onnx"
    if not os.path.exists(MODIFIED_MODEL):
        onnx_model = onnx.load(MODEL)

        for initializer in onnx_model.graph.initializer:
            if initializer.name == "Parameter193":
                # Dense layer weights (256 x 10)
                weights = np.array(initializer.float_data).reshape((256, 10))
                u, s, vh = scipy.sparse.linalg.svds(weights, k=k)
                initializer.float_data[:] = (u @ np.diag(s) @ vh).reshape((2560,))

        onnx.save(onnx_model, MODIFIED_MODEL)

    new_session = ort.InferenceSession(MODIFIED_MODEL)

    for i in range(NUM_TEST_DATA_IMAGES):
        result = new_session.run(["Plus214_Output_0"], {"Input3": TEST_DATA_IMAGES[i]})
        # result = session.run(None, {"Input3": TEST_DATA_IMAGES[i]})
        predicted_values[i] = np.argmax(result)

    acc = np.mean(predicted_values == TEST_DATA_LABELS)
    acc_list.append(acc)
    print(f"New accuracy for k = {k}: {acc}")

# Plot the results.
ax = sns.lineplot(x=K_VALUES, y=acc_list, label="Accuracy")
for x, y in zip(K_VALUES, acc_list):
    ax.text(x, y, str(y), ha='center', va='bottom')
ax.set(title="Accuracy v. Rank")
ax.set(xlabel="Rank (k)")
ax.set(ylabel="Accuracy")
plt.savefig("MNIST_accuracy_vs_rank.png")
"""

# Calculate space saved 
for k in K_VALUES:
    og_number_of_param = 256 * 10
    new_number_of_param = 256 * k + k + 10 * k
    number_saved = og_number_of_param - new_number_of_param
    percent_saved_layer = number_saved / og_number_of_param * 100
    percent_saved_total = number_saved / 5998 * 100
    print(f"k = {k}: new number of parameters in this weight:{new_number_of_param}")
    print(f"Number of parameters saved: {number_saved}, {round(percent_saved_layer, 2)}% of current matrix, {round(percent_saved_total, 2)}% total\n")