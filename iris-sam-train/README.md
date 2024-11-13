## Requirements
- Download SAM's [checkpoints](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints)
 
First, we need to make a conda environment with the following command:
```
conda create -n iris-sam python=3.8
conda activate iris-sam
```
Then, we need to install the requirements:
```
pip install -r requirements.txt
```

## Dataset
The dataset is provided in `data` directory along with images and masks. The images are in `.jpg` format.

## Training
To train the model, run the following command:
```
python sam_train_losses.py --data_dir data  --epochs 100 --loss focal --lr 0.0001 --save_dir checkpoints/ND


Model checkpoints will be saved in `save_dir` (default: `checkpoints/ft`). The program will save two checkpoints, one with optimizer and one without optimizer. The optimizer checkpoint is used for resuming training.
The checkpoint without optimizer (`model.pt`) is used for inference. We already provide a fine-tuned checkpoint in `google drive` and suggest to use model.pt.
```
## Testing
We also provide a script to test the model on the val set. To test the model, run the following command:

```
python sam_test.py --data_dir data --pretrained_model *path_to_model/model.pt --save_dir *path_to_output/ND_test
```

## Inference
To run inference, we need to provide bbox coordinates for the segmentation region in the image (automated). This is how SAM works,i.e. it requires a prompt along with image for prediction. In our case, the prompt is the bbox coordinates. We provide a script to run inference on a single image. To run inference, run the following command:

```     
python sam_infer.py --image_path {PATH_TO_IMAGE_OR_DIR} --pretrained_model weights/model.pt --save_dir results
```

when  you run the above command, an image window will be opened. You will be asked to draw (click) the top left and bottom right coordinates of the bbox using mouse pointer.  Results will be saved in `save_dir` (default: `outputs/results`). The results include the overlay of predicted and ground truth masks. 

The inference command can either take a single image or a directory containing multiple images as input. In case of directory, also provide the extension of the images. For example, if the images are in `.jpg` format, run the following command:

```
python sam_infer.py --image_path {PATH_TO_IMAGE_DIR} -extension jpg --pretrained_model weights/model.pt --save_dir results
```


If you get errors with cv2 while plotting bboxes on the image, uninstall and install OpenCV as follows:
```
pip uninstall opencv-python-headless -y 
pip install opencv-python --upgrade
```

## Training with multiple classes dataset
This is when we have non-binary masks, with multiple classes.
for example in the CASIA dataset, we have created 5 classes using SAM: , iris, pupil, sclera,  top_eyelash, bottom_eyelash.
To train the model on a dataset with multiple classes, run the following command:
```
pyhon sam_train_losses.py --data_dir {data_path} --save_dir {path_to_save_outputs}  --multi_cls --num_classes {num_classes}
```
where `data_path` is the path to the dataset, `path_to_save_outputs` is the path to save the model checkpoints, `num_classes` is the number of classes in the dataset.
# ex
 python sam_train_losses.py --data_dir data/casia_multi_cls/ --save_dir checkpoints/focal_casia_multi_cls  --multi_cls --num_classes 5
```

## Testing with multiple classes dataset
To test the model on a dataset with multiple classes, run the following command:
```
python sam_test.py --data_dir {data_path} --pretrained_model {path_to_model/model.pt} --save_dir {path_to_output} --multi_cls --num_classes {num_classes}
```
where `data_path` is the path to the dataset, `path_to_model/model.pt` is the path to the model checkpoint, `path_to_output` is the path to save the outputs, `num_classes` is the number of classes in the dataset.
# ex
```
python sam_test.py --data_dir data/casia_multi_cls_correct_split/ --save_dir output/multi_cls_ccorrect --pretrained_model checkpoints/focal_casia_multi_cls_correct_split/gamma_2.0/model.pt --multi_cls --num_classes 5
```
python sam_test.py --data_dir data/casia_multi_cls/ --save_dir output/multi_cls_correct --pretrained_model checkpoints/focal_casia_multi_cls/gamma_2.0/model.pt --multi_cls --num_classes 5



ssh-keygen -t ed25519 -C "parisa.farmanifard@gmail.com"
