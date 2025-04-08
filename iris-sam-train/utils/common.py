import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

def save_model(model, path, epoch=None, optimizer=None, data_parallel=False, make_dict=True):
    if data_parallel:
        state_dict = {
            'model': model.module.state_dict(),
        }
    else:
        state_dict = {
            'model': model.state_dict(),
        }
    if make_dict:
        if optimizer is not None:
            state_dict['optimizer'] = optimizer.state_dict()
        if epoch is not None:
            state_dict['epoch'] = epoch
        
        torch.save(state_dict, path)
    else:
        torch.save(state_dict['model'], path)
    print("save model done. path=%s" % path)

def load_model(model, path, optimizer=None, data_parallel=False):
    # Load the checkpoint
    # checkpoint = torch.load(path)
    # Debug: print out the keys in the checkpoint
    # print("Checkpoint keys:", checkpoint.keys())
    state_dict = torch.load(path)
    if data_parallel:
        model.module.load_state_dict(state_dict['model'], strict=True)
    else:
        model.load_state_dict(state_dict['model'], strict=True)
    
    print("load model done. path=%s" % (path))
    
    if optimizer is not None and  'optimizer' in state_dict:
        optimizer.load_state_dict(state_dict['optimizer'])
    if 'epoch' in state_dict:
        epoch = state_dict['epoch']
        return epoch

def get_bounding_box(ground_truth_map):
    """The function below defines how to get a bounding box prompt based on the ground truth segmentation.
    This was taken from [here](https://github.com/bowang-lab/MedSAM/blob/66cf4799a9ab9a8e08428a5087e73fc21b2b61cd/train.py#L29)."""
    # get bounding box from mask
    y_indices, x_indices = np.where(ground_truth_map > 0)
    if len(y_indices) == 0 or len(x_indices) == 0:
        return np.array([0, 0, 0, 0])
    
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    
    # add perturbation to bounding box coordinates
    H, W = ground_truth_map.shape
    x_min = max(0, x_min - np.random.randint(0, 20))
    x_max = min(W, x_max + np.random.randint(0, 20))
    y_min = max(0, y_min - np.random.randint(0, 20))
    y_max = min(H, y_max + np.random.randint(0, 20))
    bbox = [x_min, y_min, x_max, y_max]

    return np.array(bbox)

def resize_and_pad_image(image, new_size):
    
    # Compute aspect ratio
    original_height, original_width = image.shape[:2]
    aspect_ratio = original_width / original_height
    
    # Determine the desired width and height
    target_height, target_width  = new_size
    target_ratio = target_width / target_height
    
    if target_ratio > aspect_ratio:
        # The image is narrower, pad the width
        resized_width = int(target_height * aspect_ratio)
        resized_size = (resized_width, target_height)
    else:
        # The image is shorter or has equal proportions, pad the height
        resized_height = int(target_width / aspect_ratio)
        resized_size = (target_width, resized_height)
    
    # Resize the image
    resized_image = cv2.resize(image, resized_size, interpolation=cv2.INTER_CUBIC)
    # Create a new blank image with the target size
    new_shape = (target_height, target_width, image.shape[-1]) if image.ndim == 3 else (target_height, target_width)
    padded_image = np.zeros((new_shape), dtype=np.uint8)
    
    padded_image[:resized_image.shape[0], :resized_image.shape[1]] = resized_image
    
    return padded_image


def undo_padding_and_resize(padded_image, original_size):
    # Compute the aspect ratio
    original_height, original_width = original_size
    aspect_ratio = original_width / original_height
    
    # Determine the target width and height
    padded_height, padded_width = padded_image.shape[:2]
    padded_ratio = padded_width / padded_height
    
    if padded_ratio > aspect_ratio:
        # The image is narrower, pad the width
        cropped_width = int(padded_height * aspect_ratio)
        cropped_height = padded_height
    else:
        # The image is shorter or has equal proportions, pad the height
        cropped_height = int(padded_width / aspect_ratio)
        cropped_width = padded_width
    
    cropped_image = padded_image[:cropped_height, :cropped_width]
    
    # Resize the image
    restored_image = cv2.resize(cropped_image, (original_width, original_height), interpolation=cv2.INTER_AREA)

    
    return restored_image


def merge_image_mask(image, mask, alpha=0.5, color=(0, 255, 0)):
    # print(image.shape, 'img')
    # print(mask.shape, 'mask')
    mask  = undo_padding_and_resize(mask, image.shape[:2])
    mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
    
    
    # Create a mask where white pixels from the binary mask are True
    white_pixels = mask == 1
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) 
    # Apply the color overlay to the color mask
    mask[white_pixels] = color    
    image = cv2.addWeighted(image, alpha, mask, 1 - alpha, 0)
    
    return image

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([1.0])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 1.0])
    h, w = mask.shape[:2]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  


def plot_masks(image, pred_mask, img_path, save_dir, gt_mask=None, gt_box=None):
    """plot image and mask side by side"""
    
        
    name = os.path.splitext(os.path.basename(img_path))[0]
    
    _, axs = plt.subplots(1, 2, figsize=(25, 25))


    # plotting GT mask
    axs[0].imshow(image)
    if gt_mask is not None:
        show_mask(gt_mask, axs[0])
        show_box(gt_box, axs[0])
        axs[0].set_title('GT Mask', fontsize=26)
    else:
        axs[0].set_title('Image', fontsize=26)
    axs[0].axis('off')
        

    # plotting predicted mask
    axs[1].imshow(image)
    show_mask(pred_mask, axs[1])
    show_box(gt_box, axs[1])
    axs[1].set_title('Predicted Mask', fontsize=26)
    axs[1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{name}.png'), dpi=300)
    plt.close()

if __name__ =='__main__':
    
    
    # img = np.random.randint(0, 255, (256, 280, 3), dtype=np.uint8)
    # mask = np.random.randint(0, 255, (256, 280), dtype=np.uint8)
    img = cv2.imread('x.jpg')
    mask = cv2.imread('/research/iprobe-farmanif/sam_ft_final/data/train/masks/02463d834.jpg', cv2.IMREAD_GRAYSCALE)
    
    im_mask = resize_and_pad_image(mask.copy(), (340, 220))
    
    im_pad = resize_and_pad_image(img.copy(), (340, 220))
    
    print(im_pad.shape , 'im_pad')
    print(im_mask.shape, 'im_mask')
    
    undo_img =  undo_padding_and_resize(im_pad, img.shape[:2])
    undo_mask = undo_padding_and_resize(im_mask, mask.shape[:2])
    
    print(undo_img.shape, 'undo_img')
    print(undo_mask.shape, 'undo_mask') 
    
    print(np.allclose(img, undo_img), 'img')
    print(np.allclose(mask, undo_mask), 'mask')
    
    
    cv2.imwrite('img.png', undo_img)
    cv2.imwrite('mask.png', im_pad)