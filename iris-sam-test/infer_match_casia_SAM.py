import cv2
import iris
import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import yaml
import numpy as np
from tqdm import tqdm

# config for SAM pipeline
config_path = 'pipeline_sam.yaml'

config = yaml.safe_load(open(config_path, 'r'))

# Initialize the iris recognition pipeline
iris_pipeline = iris.IRISPipeline(config=config)
print("IRIS Pipeline initialized.")

# Initialize the matcher
matcher = iris.HammingDistanceMatcher()
print("Hamming Distance Matcher initialized.")

# Directory containing iris images
# data_dir = "open-iris/CASIA-v3
data_dir = '../Finetuned-SAM/data/casia_multi_cls_correct_split/test/images/'
print(f"Processing images from directory: {data_dir}")

# mask dir to get bboxes
mask_dir = '../Finetuned-SAM/data/casia_multi_cls_correct_split/test/masks/'

# where to save the results
results_dir = 'outputs/open-iris/results_correct_LR'

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

def parse_filename(filename):
    # Remove the extension and then split by underscores if present
    base_name = filename.split('.')[0]
    id_part = base_name[:-3]  # Everything except the last three characters (e.g., 'L01' or 'R01')
    eye_side_letter = base_name[-3]  # The character for 'L' or 'R'
    eye_side = 'left' if eye_side_letter == 'L' else 'right' if eye_side_letter == 'R' else None

    if eye_side is None:
        print(f"Warning: Eye side not detected correctly in filename {filename}.")
        return None, None
    return id_part, eye_side

def load_and_create_template(file_path, mask_path):
    print(f"Loading image: {file_path}")
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    masks = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    bboxes = masks_to_bboxes(masks, num_classes=5)
    
    if image is not None:
        id_part, eye_side = parse_filename(os.path.basename(file_path))
        eye_side = 'left' if eye_side == 'left' else 'right'
        output = iris_pipeline(img_data=image, eye_side=eye_side, bboxes=bboxes)
        if 'iris_template' in output and output['error'] is  None:
            return True, output['iris_template']
    return False, output['error'] #f"Failed to process image at {file_path}"

def process_image(file_path, mask_path):
    success, result = load_and_create_template(file_path, mask_path)
    
    if success:
        return file_path, result, None
    else:
        print(f"Failed to process image at {file_path}. Error: {result}")
        return file_path, None, result['error_type']

def process_images(images_dir, masks_dir):
    templates = {}
    failures = []
    # with ProcessPoolExecutor() as executor:
        
    #     file_paths = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith((".jpg", ".jpeg", ".png"))]
    #     images_names = [os.path.splitext(os.path.basename(f))[0] for f in file_paths]
    #     mask_paths = [os.path.join(masks_dir, f + '.png') for f in images_names] # replace extension with .png
    #     results = executor.map(process_image, file_paths, mask_paths)
    #     for file_path, template, error in results:
    #         if template:
    #             templates[os.path.basename(file_path)] = template
    #         else:
    #             failures.append({'Filename': file_path, 'Error': error})
        
    file_paths = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith((".jpg", ".jpeg", ".png"))]
    images_names = [os.path.splitext(os.path.basename(f))[0] for f in file_paths]
    mask_paths = [os.path.join(masks_dir, f + '.png') for f in images_names] # replace extension with .png
    
    for file_path, mask_path in zip(file_paths, mask_paths):
        results = process_image(file_path, mask_path)
        file_path, template, error = results
        if template:
            templates[os.path.basename(file_path)] = template
        else:
            failures.append({'Filename': file_path, 'Error': error})
    return templates, failures

def compare_templates(templates):
    results = []
    filenames = list(templates.keys())
    print("Comparing templates...")
    for i, file1 in enumerate(tqdm(filenames[:-1], 'Comparing')):
        for file2 in filenames[i+1:]:
            if templates[file1] and templates[file2]:
                id1, eye1 = parse_filename(file1)
                id2, eye2 = parse_filename(file2)
                
                if None in [id1, eye1, id2, eye2]:
                    continue  # Skip if any parsing resulted in None

                match_type = 'Genuine' if id1 == id2 and eye1 == eye2 else 'Imposter'
                results.append({
                    'File1': file1,
                    'File2': file2,
                    'Hamming Distance': matcher.run(templates[file1], templates[file2]),
                    'Match Type': match_type
                })
    return results

def main():
    templates, failures = process_images(data_dir, mask_dir)
    results = compare_templates(templates)
    results_df = pd.DataFrame(results)
    os.makedirs(results_dir, exist_ok=True)
    
    results_df.to_csv(os.path.join(results_dir, "CASIA_LR.csv"), index=False)
    print("Matching completed. Results saved to CASIA_LR.csv.")

    if failures:
        failures_df = pd.DataFrame(failures)
        failures_df.to_csv(os.path.join(results_dir, "CASIA_failures_LR.csv"), index=False)
        print(f"Failed cases saved to CASIA_failures_LR.csv with {len(failures)} entries.")

if __name__ == '__main__':
    main()
