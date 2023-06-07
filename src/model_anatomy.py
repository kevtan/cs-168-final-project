import numpy as np
import onnx
import pandas as pd

################################################################################
# Load the model.
################################################################################
MODEL_PATH = "../ImageNetModels/shufflenet-v2-10.onnx"
onnx_model = onnx.load(MODEL_PATH)

################################################################################
# Determine the number of weights in each initializer.
################################################################################
data = {
    "name": [],
    "size": [],
}

for initializer in onnx_model.graph.initializer:
    data["name"].append(initializer.name)
    data["size"].append(np.prod(initializer.dims))

df = pd.DataFrame(data=data)

################################################################################
# Print initializers in order of size.
################################################################################
df = df.sort_values(by="size", ascending=False)
df["cumsum"] = df["size"].cumsum()
df["percentage"] = df["cumsum"] / df["size"].sum()
df["cumpercentage"] = df["percentage"].cumsum()
print(df.head(10))
