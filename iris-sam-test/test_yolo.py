import cv2
import iris
import yaml



if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Run the IRIS pipeline on an image.")
    parser.add_argument("--image", type=str, help="Path to the image to run the pipeline on.")
    parser.add_argument("--eye_side", type=str, choices=["left", "right"],
                        help="Side of the eye to run the pipeline on.")
    parser.add_argument("--config", type=str, default="pipeline_yolo_sam.yaml",
                        help="Path to the pipeline configuration file.")
    
    parser.add_argument("--save_dir", type=str, default="outputs/yolo_sam",
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

    # 3. get eye side if not provided
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
    output = iris_pipeline(img_data=img_pixels, eye_side=args.eye_side)
    
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