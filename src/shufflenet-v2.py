import onnxruntime as ort
import pandas as pd
import tqdm
from PIL import Image
from torchvision import transforms

if __name__ == "__main__":
    ############################################################################
    # Read in the CLS-LOC validation blacklist file.
    ############################################################################
    blacklisted_image_indices = set()
    with open(
        "../ILSVRC2014_devkit/data/ILSVRC2014_clsloc_validation_blacklist.txt", "r"
    ) as f:
        lines = f.readlines()
        for line in lines:
            blacklisted_image_indices.add(int(line.strip()))
    assert len(blacklisted_image_indices) == 1762

    ############################################################################
    # Read in the CLS-LOC validation ground truth file.
    ############################################################################
    validation_labels = {}
    with open(
        "../ILSVRC2014_devkit/data/ILSVRC2014_clsloc_validation_ground_truth.txt", "r"
    ) as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            validation_labels[i + 1] = int(line.strip())

    ############################################################################
    # Read in the synsets files (which converts numerical classes to strings).
    #
    #        ILSVRC2014_ID                                       words
    #   0                1                    kit fox, Vulpes macrotis
    #   1                2                              English setter
    #   2                3                              Siberian husky
    #   3                4                          Australian terrier
    #   4                5  English springer, English springer spaniel
    #   ..             ...                                         ...
    #   995            996                                  coffee mug
    #   996            997        rubber eraser, rubber, pencil eraser
    #   997            998                                       stole
    #   998            999                                   carbonara
    #   999           1000                                    dumbbell
    #
    ############################################################################
    synsets = pd.read_csv("../ILSVRC2014_devkit/data/synsets.csv")

    ############################################################################
    # Control the number of images considered (from 1 to 50,000).
    ############################################################################
    N_IMAGES = 100

    ############################################################################
    # Load the pre-trained model and perform inference.
    ############################################################################
    MODEL_FILE = "../ImageNetModels/shufflenet-v2-10.onnx"

    ort_session = ort.InferenceSession(MODEL_FILE)

    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Loop through the (1-indexed) images.
    for i in range(1, N_IMAGES + 1):
        # Skip blacklisted images.
        if i in blacklisted_image_indices:
            continue

        # Read in the image.
        image_filename = f"../ILSVRC2012_img_val/ILSVRC2012_val_{i:08d}.JPEG"
        input_image = Image.open(image_filename)

        # Skip grayscale images.
        if input_image.mode != "RGB":
            continue

        # Perform preprocessing.
        input_tensor = preprocess(input_image)

        # Add batch dimension.
        input_batch = input_tensor.unsqueeze(0)

        # Run the model.
        output = ort_session.run(None, {"input": input_batch.numpy()})[0].squeeze()

        # Get the top-5 predictions.
        top_5_predictions = output.argsort()[-5:][::-1]
