from segment_anything import  sam_model_registry
import pickle
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
from tqdm import tqdm
import torch
from torch.optim import Adam
from torch.optim import SGD
from torch.utils.data import DataLoader
import argparse
import torch.nn as nn
from torch.nn.functional import threshold, normalize
from torch.utils import tensorboard
import numpy as np
from dataset import SAMDataset, SAMDatasetMultiClass
from utils.common import save_model, load_model


def dice_loss(pred, target, smooth=1.):
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def iou_loss(pred, target):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return 1 - (intersection + 1) / (union + 1)

def focal_loss(pred, target, alpha=0.8, gamma=2):
    BCE_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    pt = torch.exp(-BCE_loss)
    F_loss = alpha * (1-pt)**gamma * BCE_loss
    return F_loss.mean()


def train(args):
    # model_type = 'vit_b'
    model_type = 'vit_h'
    # checkpoint = 'weights/sam_vit_b_01ec64.pth'
    checkpoint = 'weights/sam_vit_h_4b8939.pth'
    
    sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    
    sam_model.to(device)
    sam_model.train()
    
    if args.multi_cls:
        train_dataset = SAMDatasetMultiClass(args.data_dir, sam_model.image_encoder.img_size, out_mask_shape=256, split='train', num_classes=args.num_classes)
    else:
        train_dataset = SAMDataset(args.data_dir, sam_model.image_encoder.img_size, out_mask_shape=256, split='train')

    # val_dataset = SAMDataset(args.data_dir, processor=processor, split='val')
    

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    # val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    # Initialize model
    # make sure we only compute gradients for mask decoder
    for name, param in sam_model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)
    
    
    # Note: Hyperparameter tuning could improve performance here
    optimizer = SGD(sam_model.mask_decoder.parameters(), lr=args.lr, weight_decay=0)

    loss_functions = {
    'mse': torch.nn.MSELoss(),
    'dice': dice_loss,
    'iou': iou_loss,
    'focal': lambda pred, target: focal_loss(pred, target, gamma=args.gamma),
    # Add other losses here
    }

    if args.loss not in loss_functions:
        raise ValueError(f"Loss {args.loss} not recognized. Available options: {list(loss_functions.keys())}")

    loss_fn = loss_functions[args.loss]

    
    if args.pretrained_model:
        cur_epoch = load_model(sam_model, args.pretrained_model, optimizer)
        start_epoch = cur_epoch + 1
    else:
        start_epoch = 0

    os.makedirs(args.save_dir, exist_ok=True)
    summary = tensorboard.SummaryWriter(args.save_dir)

    print('starting training from epoch', start_epoch)
    
    epoch_loss_history = []

    for epoch in range(start_epoch, args.epochs):
        epoch_losses = []
        
        pbar = tqdm(train_dataloader, desc=f'epoch {epoch+1}/{args.epochs}; batch: {0}/{len(train_dataloader)}; loss: {0}')
        for idx, batch in enumerate(pbar):
            try:
                batch = next(iter(train_dataloader))  
                # forward pass
                input_images = sam_model.preprocess(batch['image'].to(device))
                prompt_boxes = batch['input_boxes'].to(device)
                # input_size = batch['input_size'][0]
                input_size = tuple(input_images.shape[-2:])
                original_image_size = batch['original_image_size'][0].cpu().numpy().tolist()
            
                # No grad here as we don't want to optimise the encoders
                with torch.no_grad():
                    image_embedding = sam_model.image_encoder(input_images)
                    boxes_torch = torch.as_tensor(prompt_boxes, dtype=torch.float, device=device)
                    
                    se_list = []
                    de_list = []
                    # we may have multiple boxes per image, loop through them
                    for i in range(boxes_torch.shape[1]):
                        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                            points=None,
                            boxes=boxes_torch[:, i, :],
                            masks=None,
                        )
                        se_list.append(sparse_embeddings)
                        de_list.append(dense_embeddings)
                
                    
                pred_masks_list = []
                iou_predictions_list = []
                # we may have multiple boxes per image, loop through them
                for i in range(len(se_list)):
                    pred_masks, iou_predictions = sam_model.mask_decoder(
                        image_embeddings=image_embedding,
                        image_pe=sam_model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=se_list[i],
                        dense_prompt_embeddings=de_list[i],
                        multimask_output=False,
                    )
                    pred_masks_list.append(pred_masks)
                    iou_predictions_list.append(iou_predictions)
                    
               
                
                # conver the list of tensors to a single tensor
                pred_masks = torch.cat(pred_masks_list, dim=1)
                iou_predictions = torch.cat(iou_predictions_list, dim=1)
                
               
                # upscaled_masks = sam_model.postprocess_masks(low_res_masks, input_size, original_image_size).to(device)
                # binary_mask = normalize(threshold(upscaled_masks, 0.0, 0))

                ground_truth_masks = batch["ground_truth_mask"].to(device)
                # gt_mask_resized = ground_truth_masks.unsqueeze(1).to(device)
                gt_binary_mask = torch.as_tensor(ground_truth_masks > 0, dtype=torch.float32)#.unsqueeze(1)
                loss = loss_fn(pred_masks, gt_binary_mask)
                # print(loss.item()  )

                # backward pass (compute gradients of parameters w.r.t. loss)
                optimizer.zero_grad()
                loss.backward()

                # optimize
                optimizer.step()
                epoch_losses.append(loss.item())
                pbar.set_description(f'epoch {epoch+1}/{args.epochs}; batch: {idx+1}/{len(train_dataloader)}; loss: {loss.item():.6f}')
            except Exception as e:
                print(f"Error encountered in batch {idx}: {e}")
                # print stack trace
                import traceback
                traceback.print_exc()
                exit()
                continue
                
            cur_step = epoch * len(train_dataloader) + idx + 1
            summary.add_scalar('loss/step', loss.item(), cur_step)

        loss = np.mean(epoch_losses)
        epoch_loss_history.append(loss)
        print(f'EPOCH: {epoch+1}/{args.epochs}, loss: {loss:.6f}')
        summary.add_scalar('loss/epoch', loss, epoch+1)
        
        # Save losses after every epoch
        with open(os.path.join(args.save_dir, 'epoch_loss_history.pkl'), 'wb') as f:
             pickle.dump(epoch_loss_history, f)
        save_path = os.path.join(args.save_dir, f"model_with_optim.pt")
        
        # save with optimizer, useful for fine-tuning
        save_model(sam_model, save_path, epoch=epoch, optimizer=optimizer, data_parallel=False)
        
        # save model only, can be  loaded directly to predict
        save_path = os.path.join(args.save_dir, f"model.pt")
        save_model(sam_model, save_path, data_parallel=False, make_dict=False)
    # save losses
    save_path = os.path.join(args.save_dir, 'epoch_loss_history.pkl')
    with open(save_path, 'wb') as f:
         pickle.dump(epoch_loss_history, f)
    print(f"Saved epoch loss history to {save_path}")

    return epoch_loss_history
 
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('SAM model training')
    
    parser.add_argument('--data_dir', type=str, default='data', help='path to data directory')
    parser.add_argument('--save_dir', type=str, default='checkpoints/focal_casia', help='path to save directory')
    parser.add_argument('--pretrained_model', type=str, default='', help='path to pretrained model')
    parser.add_argument('--gamma', type=float, default=2, help='gamma value for focal loss')
    parser.add_argument('--loss', type=str, default='focal', help='Loss function to use. Options: mse, dice, iou, focal.')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size') # doesnot work with batch size > 1
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--multi_cls', action='store_true', help='use multi-class segmentation') # this just means our masks are not binary
    parser.add_argument('--num_classes', type=int, default=5, help='number of classes for multi-class segmentation' \
                        'only used if --multi_cls is set. [default: 5], (iris, pupil, sclera, top_lashes, bottom_lashes)')
                        
    

    args = parser.parse_args()
    # epoch_loss_history = train(args)
    original_save_dir = args.save_dir
    loss_histories ={}
    
# Setting font for plots
font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 16,
        }

for gamma in [2.0]:
    args.gamma = gamma
    args.save_dir = os.path.join(original_save_dir, f"gamma_{gamma}")
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Train with the current gamma value
    losses = train(args)  # Assuming your train function returns a list of losses
    loss_histories[gamma] = losses
    
    # Plot individual gamma
    plt.figure(figsize=(10, 6))
    plt.plot(losses, linewidth=2)
    plt.title(f'Training Loss Curve for Gamma={gamma}', fontdict=font)
    plt.xlabel('Epoch', fontdict=font)
    plt.ylabel('Loss', fontdict=font)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, f'loss_curve_gamma_{gamma}.png'), dpi=300)
    plt.show()

# Combined loss plot for all gammas
plt.figure(figsize=(12, 7))
for gamma, losses in loss_histories.items():
    plt.plot(losses, label=f'Gamma={gamma}', linewidth=2)
plt.title('Training Loss Curve for Different Gammas', fontdict=font)
plt.xlabel('Epoch', fontdict=font)
plt.ylabel('Loss', fontdict=font)
plt.legend(loc='best', fontsize='medium')
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(original_save_dir, 'combined_loss_curve.png'), dpi=300)
plt.show()
