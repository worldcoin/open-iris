import cv2
import iris
import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import yaml
import numpy as np
from tqdm import tqdm

# config for SAM pipeline
config_path = 'pipeline_yolo_sam.yaml'

config = yaml.safe_load(open(config_path, 'r'))

# Initialize the iris recognition pipeline
iris_pipeline = iris.IRISPipeline(config=config)
print("IRIS Pipeline initialized.")

# Initialize the matcher
matcher = iris.HammingDistanceMatcher()
print("Hamming Distance Matcher initialized.")

# Directory containing iris images
# data_dir = "open-iris/CASIA-v3
data_dir = '../data/casia_multi_cls_correct_split/test/images/'
print(f"Processing images from directory: {data_dir}")


# where to save the results
results_dir = 'outputs/open-iris/results_correct_LR_YOLO'



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

def load_and_create_template(file_path):
    print(f"Loading image: {file_path}")
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    
    if image is not None:
        id_part, eye_side = parse_filename(os.path.basename(file_path))
        eye_side = 'left' if eye_side == 'left' else 'right'
        output = iris_pipeline(img_data=image, eye_side=eye_side)
        if 'iris_template' in output and output['error'] is  None:
            return True, output['iris_template']
    return False, output['error'] #f"Failed to process image at {file_path}"

def process_image(file_path):
    success, result = load_and_create_template(file_path)
    
    if success:
        return file_path, result, None
    else:
        print(f"Failed to process image at {file_path}. Error: {result}")
        return file_path, None, result['error_type']

def process_images(images_dir):
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
    
    for f_idx, file_path in enumerate(file_paths):
        results = process_image(file_path)
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
    templates, failures = process_images(data_dir)
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
