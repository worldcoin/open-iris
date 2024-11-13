import cv2
import iris
import yaml
import numpy as np


def get_bounding_box(ground_truth_map, add_noise=True):
    """The function below defines how to get a bounding box prompt based on the ground truth segmentation.
    This was taken from [here](https://github.com/bowang-lab/MedSAM/blob/66cf4799a9ab9a8e08428a5087e73fc21b2b61cd/train.py#L29)."""
    # get bounding box from mask
    y_indices, x_indices = np.where(ground_truth_map > 0)
    if len(y_indices) == 0 or len(x_indices) == 0:
        return np.array([0, 0, 0, 0])
    
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    if add_noise:
        # add perturbation to bounding box coordinates
        H, W = ground_truth_map.shape
        x_min = max(0, x_min - np.random.randint(0, 20))
        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))
    bbox = [x_min, y_min, x_max, y_max]

    return np.array(bbox)

def masks_to_bboxes(masks, num_classes):
    """
    A function to convert a mask to bounding boxes used as input to SAM model
    """
    masks_list = []
    for i in range(num_classes):
        mask = (masks == i+1).astype(np.uint8) * 255
        masks_list.append(mask)
    
    bboxes = []
    for gt_mask in masks_list:
        bbox = get_bounding_box(gt_mask)
        bboxes.append(bbox)
    bboxes = np.array(bboxes)
    
    return bboxes

if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Run the IRIS pipeline on an image.")
    parser.add_argument("--image", type=str, help="Path to the image to run the pipeline on.")
    parser.add_argument("--mask", type=str, help="Path to the ground truth mask."
                        "We use the masks to get the bounding box prompt only.")
    parser.add_argument("--eye_side", type=str, choices=["left", "right"],
                        help="Side of the eye to run the pipeline on.")
    parser.add_argument("--config", type=str, default="pipeline_sam.yaml",
                        help="Path to the pipeline configuration file.")
    
    parser.add_argument("--save_dir", type=str, default="outputs",
                        help="Path to save the output data")
    
    args = parser.parse_args()
    
    
    # 1. Create IRISPipeline object
    with open(args.config, "r") as f: 
        conf = yaml.safe_load(f)
    print("Configuration loaded.")
    print(conf)
    iris_pipeline = iris.IRISPipeline(config=conf)
    print("Pipeline created.")
    
    # 2. Load IR image of an eye
    img_pixels = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)

    # 3. Load the ground truth mask and get the bounding box prompt
    masks = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    
    bboxes = masks_to_bboxes(masks, num_classes=5) # 5 classes: iris, pupil, scleara, top_eyelashes, bottom_eyelashes
    # get eye side if not provided
    if not args.eye_side:
        if "L" in args.image:
            args.eye_side = "left"
        elif "R" in args.image:
            args.eye_side = "right"
        else:
            raise ValueError("Please provide the eye side.")

    # 4. Perform inference
    # Options for the `eye_side` argument are: ["left", "right"]
    print("Running the pipeline...")
    output = iris_pipeline(img_data=img_pixels, eye_side=args.eye_side, bboxes=bboxes)
    
    if output["error"]:
        print(output["error"])
        exit(1)
    
    # 5. Save the output
    image_name = os.path.splitext(os.path.basename(args.image))[0]
    ouput_dir = os.path.join(args.save_dir, image_name)
    os.makedirs(ouput_dir, exist_ok=True)

    for key, value in output.items():
        if key == "error":
            continue
        
        with open(os.path.join(ouput_dir, f"{key}.txt"), "w") as f:
            f.write(str(value))
    print(f"Output saved in {ouput_dir}")