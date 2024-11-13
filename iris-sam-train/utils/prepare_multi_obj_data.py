# a script to prepare multi-object data for training
# is run only once to prepare the data
# it copies only the images and masks that are in the selected file


import os
import shutil
import argparse

def prepare(data_dir, save_dir):
    
    images_dir = os.path.join(data_dir, 'images')
    masks_dir = os.path.join(data_dir, 'processed_masks')
    
    train_file = os.path.join(data_dir, 'train.txt')
    val_file = os.path.join(data_dir, 'val.txt')
    
    # images in this file were manually selected to have good masks for all classes (msks generated using SAM)
    selected_file = os.path.join(data_dir, 'selected.txt')
    
    
    with open(selected_file, 'r') as f:
        selected_images = [line.strip() for line in f]
        selected_images = set(selected_images)
    
    with open(train_file, 'r') as f:
        train_images = [line.strip() for line in f]
        train_images = set(train_images)
    
    with open(val_file, 'r') as f:
        val_images = [line.strip() for line in f]
        val_images = set(val_images)
    
    # only keep images in selected file for training and validation
    train_images = train_images.intersection(selected_images)
    val_images = val_images.intersection(selected_images)
    
    print(f'{len(train_images)} training images')
    print(f'{len(val_images)} validation images')
    
    # copy images and masks to train directories
    train_dir = os.path.join(save_dir, 'train')
    train_images_dir = os.path.join(train_dir, 'images')
    train_masks_dir = os.path.join(train_dir, 'masks')
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_masks_dir, exist_ok=True)

    for image in train_images:
        image_path = os.path.join(images_dir, image)
        mask_path = os.path.join(masks_dir, image.replace('.jpg', '.png'))
        shutil.copy(image_path, train_images_dir)
        shutil.copy(mask_path, train_masks_dir)
    
    
    # copy images and masks to test directories
    test_dir = os.path.join(save_dir, 'test')
    test_images_dir = os.path.join(test_dir, 'images')
    test_masks_dir = os.path.join(test_dir, 'masks')
    os.makedirs(test_images_dir, exist_ok=True)
    os.makedirs(test_masks_dir, exist_ok=True)
    
    for image in val_images:
        image_path = os.path.join(images_dir, image)
        mask_path = os.path.join(masks_dir, image.replace('.jpg', '.png'))
        shutil.copy(image_path, test_images_dir)
        shutil.copy(mask_path, test_masks_dir)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Prepare multi-object data')
    parser.add_argument('--input_data_dir', type=str, default='', help='path to data directory')
    
    parser.add_argument('--save_dir', type=str, default='data', help='path to save directory')
    

    args = parser.parse_args()
    
    data_dir = args.input_data_dir
    save_dir = args.save_dir
    
    prepare(data_dir, save_dir)
    
    


