import argparse
import logging
import os

import cv2
from iris.pipelines.iris_pipeline import IRISPipeline

if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
    logging.info("Run iris pipeline inference script STARTED")

    parser = argparse.ArgumentParser("Perform IRISPipeline inference for a given image.")
    parser.add_argument(
        "-i",
        "--in_img",
        type=str,
        default=os.path.join("tests", "e2e_tests", "pipelines", "mocks", "inputs", "anonymized.png"),
    )
    args = parser.parse_args()

    iris_pipeline = IRISPipeline()

    img_data = cv2.imread(args.in_img, cv2.IMREAD_GRAYSCALE)

    out = iris_pipeline(img_data, "right")

    logging.info("Run iris pipeline inference script FINISHED")
