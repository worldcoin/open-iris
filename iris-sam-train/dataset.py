# ND-iris-0405 dataset
import numpy as np
import os
import torch
from torch.utils.data import Dataset
import cv2
# from transformers import  SamImageProcessor
from segment_anything.utils.transforms import ResizeLongestSide

from utils.common import get_bounding_box, resize_and_pad_image, undo_padding_and_resize

class SAMDataset(Dataset):      
    def __init__(self, data_dir, img_size, out_mask_shape=256, split='train', transform=None, return_index=False, identity_range=None):
        
        self.data_dir = data_dir
        self.img_size = img_size
        self.transform = transform  # Added transform as an attribute
        self.return_index = return_index
        self.out_mask_shape = out_mask_shape
        if out_mask_shape > 0:
            self.mask_transform = ResizeLongestSide(out_mask_shape)
        # Directories for images and masks
        image_dir = os.path.join(data_dir, split, "images")
        mask_dir = os.path.join(data_dir, split, "masks")
        
        if split == "train":
            self.transform = ResizeLongestSide(img_size)
        else:
            self.transform = None
        
        # Get list of image names from the image directory
        image_names = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        
        self.images = []
        self.masks = []
        
        for image_name in image_names:
            image_path = os.path.join(image_dir, image_name)
            mask_path = os.path.join(mask_dir, image_name)  # No renaming needed since both have .jpg extension
            
            if not os.path.exists(mask_path):
                # try png
                mask_path = os.path.join(mask_dir, image_name.replace('.jpg', '.png'))
                if not os.path.exists(mask_path):
                    print(f"Warning: Corresponding mask for image {image_name} not found. Skipping...")
                    continue

            self.images.append(image_path)
            self.masks.append(mask_path)
        """_summary_
        Convert to numpy arrays for possible efficiency reasons (can be kept as lists if preferred)
        """
        self.images = np.array(self.images)
        self.masks = np.array(self.masks)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading image {image_path}. Skipping index {idx}.")
            return None  # Don't raise an error, just return None
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except cv2.error as e:
            raise ValueError(f"Error converting image {image_path} to RGB: {e}")

        original_image_size = image.shape[:2]

        mask_path = self.masks[idx]
        ground_truth_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if ground_truth_mask is None:
            raise ValueError(f"Error loading mask for image {image_path}. Skipping index {idx}.")

        
        box = get_bounding_box(ground_truth_mask)
        
        if self.transform is not None: # do not do this for validation
            image = self.transform.apply_image(image)
            image = torch.as_tensor(image)
            image = image.permute(2, 0, 1).contiguous()
            input_size = tuple(image.shape[-2:])
        
            box = self.transform.apply_boxes(box, original_image_size)
        else:
            input_size = tuple(image.shape[:2])
        
        if self.out_mask_shape > 0:
            ground_truth_mask = self.mask_transform.apply_image(ground_truth_mask)
            h, w = ground_truth_mask.shape[:2]
            padh = self.out_mask_shape - h
            padw = self.out_mask_shape - w
            ground_truth_mask = np.pad(ground_truth_mask, ((0, padh), (0, padw)), mode='constant', constant_values=0)
            ground_truth_mask[ground_truth_mask > 117] = 255
            ground_truth_mask[ground_truth_mask <= 117] = 0
        
        # expand dims of ground_truth_mask
        ground_truth_mask = np.expand_dims(ground_truth_mask, axis=0)

        inputs = {}
        inputs['image'] = image
        inputs['input_boxes'] = box
        inputs['input_size'] = input_size
        inputs['original_image_size'] = np.array([original_image_size[0], original_image_size[1]])
        inputs["ground_truth_mask"] = ground_truth_mask
        
        if self.return_index:
            inputs["index"] = idx

        return inputs

class SAMDatasetMultiClass(SAMDataset):
    
    def __init__(self, data_dir, img_size, out_mask_shape=256, split='train', transform=None,
                 return_index=False, identity_range=None, num_classes=5):
        
        self.num_classes = num_classes
        super().__init__(data_dir, img_size, out_mask_shape, split, transform, return_index, identity_range)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading image {image_path}. Skipping index {idx}.")
            return None  # Don't raise an error, just return None
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except cv2.error as e:
            raise ValueError(f"Error converting image {image_path} to RGB: {e}")

        original_image_size = image.shape[:2]

        mask_path = self.masks[idx]
        ground_truth_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if ground_truth_mask is None:
            raise ValueError(f"Error loading mask for image {image_path}. Skipping index {idx}.")

        # convert to multi-class mask
        # masks are saved in one image with each class having a unique value
        # make a list of masks for each class
        ground_truth_masks = []
        for i in range(self.num_classes):
            mask = (ground_truth_mask == i+1).astype(np.uint8) * 255
            ground_truth_masks.append(mask)
        
        boxes = []
        for gt_mask in ground_truth_masks:
            box = get_bounding_box(gt_mask)
            boxes.append(box)
        boxes = np.array(boxes)
                
        if self.transform is not None: # do not do this for validation
            image = self.transform.apply_image(image)
            image = torch.as_tensor(image)
            image = image.permute(2, 0, 1).contiguous()
            input_size = tuple(image.shape[-2:])
        
            boxes = self.transform.apply_boxes(boxes, original_image_size)
        else:
            input_size = tuple(image.shape[:2])
        
        if self.out_mask_shape > 0:
            for i, gt_mask in enumerate(ground_truth_masks):
                gt_mask = self.mask_transform.apply_image(gt_mask)
                h, w = gt_mask.shape[:2]
                padh = self.out_mask_shape - h
                padw = self.out_mask_shape - w
                gt_mask = np.pad(gt_mask, ((0, padh), (0, padw)), mode='constant', constant_values=0)
                gt_mask[gt_mask > 117] = 255
                gt_mask[gt_mask <= 117] = 0
                ground_truth_masks[i] = gt_mask
        
        # make numpy array
        ground_truth_masks = np.array(ground_truth_masks)      

        inputs = {}
        inputs['image'] = image
        inputs['input_boxes'] = boxes
        inputs['input_size'] = input_size
        inputs['original_image_size'] = np.array([original_image_size[0], original_image_size[1]])
        inputs["ground_truth_mask"] = ground_truth_masks
        
        if self.return_index:
            inputs["index"] = idx

        return inputs



if __name__ == '__main__':
    # from transformers import SamProcessor
    from segment_anything import  sam_model_registry
    sam_model = sam_model_registry['vit_h']()#checkpoint='weights/sam_vit_h_4b8939.pth')
    # processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    dataset = SAMDataset("data/casia_multi_cls", 1024, split="train", out_mask_shape=256, return_index=False)
    # dataset = SAMDatasetMultiClass("data/casia_multi_cls", 1024, split="train", out_mask_shape=256, return_index=False, num_classes=5)
    
    # # Iterate over the dataset
    # for i in range(len(dataset)):
    #     try:
    #         item = dataset[i]
    #     except ValueError as e:
    #         print(e)  # Log the error
    #         continue  # Skip the rest of the processing for this iteration

    #     # Process the item
    #     input_image = sam_model.preprocess(item["image"][None, :, :, :])
    #     print(input_image.shape, 'input_image')

    #     mask = item["ground_truth_mask"]
    #     print(np.unique(mask), 'mask')
    #     print(mask.shape, mask.dtype, 'mask shape and dtype')
    #     break

    item = dataset[5]
    for k, v in item.items():
        if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray):
            print(k, v.shape, v.dtype)
        else:
            print(k,  v, type(v))
            
    # input_image = sam_model.preprocess(item["image"][None, :, :, :])
    # print(input_image.shape, 'input_image')
    
    # # boxes = item["input_boxes"].numpy()
    # image = item["image"].permute(1,2,0).numpy()
    # image_name = dataset.images[5]
    
    # image = (image * 255).astype(np.uint8)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # # image = cv2.imread(image_name)
    # boxes = item["input_boxes"]
    # colors = np.array([(0,0,255), (0,255,0), (255,0,0), (255,255,0), (255,0,255)], dtype=np.uint8)
    # for idx,  box in enumerate(boxes):
    #     x_min, y_min, x_max, y_max = box.astype(int)
    #     color = (int(colors[idx][0]), int(colors[idx][1]), int(colors[idx][2]))
    #     print(color, len(color))
    #     cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
    # cv2.imwrite("test_image.png", image)
    
    # mask = item["ground_truth_mask"]
    # # # mask = cv2.resize(mask, (1024, 1024))#.astype(np.uint8) * 255
    # # # mask = undo_padding_and_resize(mask, image.shape[:2]) *255
    # # # mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
    # # # image = cv2.resize(image, (320, 320))
    # # # mask = cv2.imread('S1144R02.png', cv2.IMREAD_GRAYSCALE)
    # print(np.unique(mask), 'mask unique')
    # print(mask.shape, mask.dtype, 'mask shape and dtype')
    
    
    # # # add mask to image as overlay
    # # mask = mask.repeat(3, axis=-1)
    # # mask = np.uint8(mask)*255
    # # # print(image.shape, mask.shape, image.dtype, mask.dtype)
    # # image = cv2.addWeighted(image, 0.5, mask, 0.5, 0)
    
    # # make a maked overlay
    
    # out_image = np.zeros(mask.shape[1:3] + (3,), dtype=np.uint8)
    # for idx, m in enumerate(mask):
    #     m = m.astype(bool).astype(np.uint8)
    #     m = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
    #     m = m * colors[idx]
    #     out_image = cv2.addWeighted(out_image, 0.5, m, 0.5, 0)
        
    
    # cv2.imwrite("test_mask.png", out_image)
    
    
    # cv2.imwrite("test_mask.png", mask)
    