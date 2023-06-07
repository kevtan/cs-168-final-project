import json
import os

import matplotlib.pyplot as plt
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import scipy
import seaborn as sns
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
    # Read in an alternate synsets convention.
    ############################################################################
    with open("../alternate_imagenet_classes.json") as f:
        synsets_alt = json.load(f)

    ############################################################################
    # Control the number of images considered (from 1 to 50,000).
    ############################################################################
    N_IMAGES = 5000

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

    print("Preprocessing images...")
    preprocessed_images = []
    # Loop through the (1-indexed) images.
    for i in tqdm.tqdm(range(1, N_IMAGES + 1)):
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

        # Save the preprocessed image.
        preprocessed_images.append((i, input_batch.numpy()))

    def run_inference(ort_session):
        # Inference statistics.
        n_processed = 0
        n_correct_top_1 = 0
        n_incorrect_top_1 = 0
        n_correct_top_5 = 0
        n_incorrect_top_5 = 0

        # Loop through the images.
        for i, input_batch in tqdm.tqdm(preprocessed_images):
            # Run the model.
            output = ort_session.run(None, {"input": input_batch})[0].squeeze()

            # Get the top-5 predictions.
            top_5_predictions = output.argsort()[-5:][::-1]

            # Convert the top-5 predictions to strings.
            top_5_prediction_strings = [synsets_alt[str(i)] for i in top_5_predictions]

            # Get the correct label as a string.
            correct_label = synsets.iloc[validation_labels[i] - 1]["words"]

            # Update the inference statistics.
            n_processed += 1
            if top_5_prediction_strings[0] == correct_label:
                n_correct_top_1 += 1
            else:
                n_incorrect_top_1 += 1
            if correct_label in top_5_prediction_strings:
                n_correct_top_5 += 1
            else:
                n_incorrect_top_5 += 1

        # Print the inference statistics.
        assert n_processed == n_correct_top_1 + n_incorrect_top_1
        assert n_processed == n_correct_top_5 + n_incorrect_top_5
        top1_acc = n_correct_top_1 / n_processed
        top5_acc = n_correct_top_5 / n_processed
        return top1_acc, top5_acc

    top1_acc, top5_acc = run_inference(ort_session)
    print("Original Statistics:")
    print(f"Top-1 Accuracy: {top1_acc:.4f}")
    print(f"Top-5 Accuracy: {top5_acc:.4f}")

    ############################################################################
    # Modify the model and perform inference.
    ############################################################################

    # Reduced ranks for the `fc.weight` initializer.
    K_VALUES = range(15, 506, 10)

    top1_acc_list = []
    top5_acc_list = []

    for k in K_VALUES:
        MODIFIED_MODEL = f"../ImageNetModels/shufflenet-v2-10-modified-{k}.onnx"
        if not os.path.exists(MODIFIED_MODEL):
            onnx_model = onnx.load(MODEL_FILE)

            # Modify the `fc.weight` initializer.
            for initializer in onnx_model.graph.initializer:
                if initializer.name == "fc.weight":
                    data = onnx.numpy_helper.to_array(initializer)
                    u, s, vh = scipy.sparse.linalg.svds(data, k=k)
                    new_tensor = (u @ np.diag(s) @ vh).astype(np.float32)
                    new_tensor = onnx.numpy_helper.from_array(new_tensor, "fc.weight")
                    initializer.raw_data = new_tensor.raw_data

            onnx.save(onnx_model, MODIFIED_MODEL)

        new_session = ort.InferenceSession(MODIFIED_MODEL)

        top1_acc, top5_acc = run_inference(new_session)
        print(f"Modified Statistics (k = {k}):")
        print(f"Top-1 Accuracy: {top1_acc:.4f}")
        print(f"Top-5 Accuracy: {top5_acc:.4f}")
        top1_acc_list.append(top1_acc)
        top5_acc_list.append(top5_acc)

    # Plot the results.
    ax = sns.lineplot(x=K_VALUES, y=top1_acc_list, label="Top-1 Accuracy")
    sns.lineplot(x=K_VALUES, y=top5_acc_list, label="Top-5 Accuracy", ax=ax)
    ax.set(title="Accuracy v. Rank")
    ax.set(xlabel="Rank (k)")
    ax.set(ylabel="Accuracy")
    plt.legend()
    plt.savefig("accuracy_vs_rank.png")
