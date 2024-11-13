import os
import glob
import cv2
images_path = os.path.join('data', 'masks')

save_dir = os.path.join('data', 'masks_jpg')

image_files = glob.glob(os.path.join(images_path, '*.png'))

if len(image_files) == 0:
    print('No masks found in the masks folder')
    exit()

os.makedirs(save_dir, exist_ok=True)


for image in image_files:
    img_name = os.path.basename(image)
    jpg_name = img_name.replace('.png', '.jpg')
    image = cv2.imread(image)
    save_path = os.path.join(save_dir, jpg_name)
    cv2.imwrite(save_path, image)
print('Images copied to data/images_fixed')
