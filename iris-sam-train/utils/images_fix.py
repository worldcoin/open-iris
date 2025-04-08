import os
import glob
import shutil

images_path = os.path.join('data', 'images')

save_dir = os.path.join('data', 'images_fixed')

image_files = glob.glob(os.path.join(images_path, '**', '*.jpg'), recursive=True)

if len(image_files) == 0:
    print('No images found in the images folder')
    exit()

os.makedirs(save_dir, exist_ok=True)


for image in image_files:
    img_name = os.path.basename(image)
    save_path = os.path.join(save_dir, img_name)
    shutil.copy(image, save_path)
print('Images copied to data/images_fixed')
