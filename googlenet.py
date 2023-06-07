import imageio.v2 as imageio
import numpy as np
import onnx
import onnxruntime as ort
import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms


if __name__ == "__main__":
    image = np.random.randint(0, 256, size=(244, 244, 3), dtype=np.uint8)

    # Plot the image
    plt.imshow(image)
    plt.axis('off')  # Turn off axis labels
    plt.savefig("test.png")

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

    ############################################################################
    # Run inference on the validation set to get baseline accuracy
    ############################################################################
    VALIDATION_LABEL_FILE = (
        "ILSVRC2014_devkit/data/ILSVRC2014_clsloc_validation_ground_truth.txt"
    )
    validation_labels = np.genfromtxt(VALIDATION_LABEL_FILE, dtype=int)
    N = 30

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    print("Preprocessing images...")
    preprocessed_images = []
    for i in tqdm.tqdm(range(N)):
        input_image = Image.open(f"ILSVRC2012_img_val/ILSVRC2012_val_{i+1:08d}.JPEG")
        input_tensor = preprocess(input_image) * 255
        input_tensor = input_tensor.detach().cpu().numpy()
        input_tensor[0, :, :] -= 123.68
        input_tensor[1, :, :] -= 116.779
        input_tensor[2, :, :] -= 103.939
        # plt.imshow(input_tensor.copy().transpose(1, 2, 0))
        # plt.axis('off')  # Turn off axis labels
        # plt.savefig(f"test{i}.png")
        input_tensor[[0,1,2],:,:] = input_tensor[[2,1,0],:,:]
        input_batch = input_tensor.reshape((1, *input_tensor.shape))
        preprocessed_images.append(input_batch)
    print("Done preprocessing.")

    session = ort.InferenceSession(MODEL_PATH)

    predicted_values = np.zeros(N)
    print("Running inference...")
    for i in tqdm.tqdm(range(N)):
        result = session.run(["prob_1"], {"data_0": preprocessed_images[i]})[0]
        predicted_values[i] = np.argmax(result)

    print(f"Accuracy is {np.mean(predicted_values == validation_labels[:N])}")
