import os
import random

random.seed(42)

data_dir = 'data'
images_dir = os.path.join(data_dir, 'images')
masks_dir = os.path.join(data_dir, 'masks')
train_val_split = 0.2


image_files = [f.name for f in  os.scandir(images_dir) if f.name.endswith('.jpg')]
mask_files = [f.name for f in  os.scandir(masks_dir) if f.name.endswith('.jpg')]
mask_files = set(mask_files)

print(len(image_files), 'total images')

print(len(image_files), 'valid images')

random.shuffle(image_files)

num_train_files = int(len(image_files) * (1 - train_val_split))
train_files = image_files[:num_train_files]
val_files = image_files[num_train_files:]

print(len(train_files), 'training images')
print(len(val_files), 'validation images')
with open(os.path.join(data_dir, 'train.txt'), 'w') as f:
    f.write('\n'.join(train_files))

with open(os.path.join(data_dir, 'val.txt'), 'w') as f:
    f.write('\n'.join(val_files))