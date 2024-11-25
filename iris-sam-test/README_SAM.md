# OPEN-IRIS with SAM

## installation


** if you have preveiously tried to install the environment, you should remove it by running the following command:
```bash
conda env remove -n iris_dev
```

Create a new conda environment and install the required packages for the SAM model by running the following commands:
```bash
conda create -n iris_dev python=3.8
conda activate iris_dev
IRIS_ENV=DEV pip install -e .
pip install -r requirements/sam.txt
```

After successfully installing `iris`, verify your installation by attempting to import.
```bash
python3 -c "import iris; print(iris.__version__)"
```


## Using the SAM model with OPEN-IRIS

1. We will need a pretrained SAM model on 5 classes of eye parts. This model is trained using Iris-SAM repository. take the model and put it in `assets/sam_pretrained` directory.

2. We need to provide a config file that defines the Iris pipeline. An example config file is provided in `pipeline_sam.yaml`. Make sure to provide the correct model verison (vit_h, vit_b) and pretrained model path in the config file.

3. Run the following command to start the OPEN-IRIS with SAM model:
```bash
python test.py --config <path_to_config_file> --image <path_to_image> --mask <path_to_mask> --save_dir <path_to_output> --eye_side <left/right>

# example
python test.py --config pipeline_sam.yaml --image data/images/S1001L01.jpg --mask data/masks/S1001L01.png --save_dir output/ --eye_side left
```

## Using SAM and YOLO together with OPEN-IRIS
1. We need the SAM model as abvoe

2. We need a pretrained YOLO-v8 model on the iris dataset. This model is trained using `yolo_det` repository. Take the `best.pt` model and put it in `assets/yolo_pretrained` directory.

3. We need to provide a config file that defines the Iris pipeline. An example config file is provided in `pipeline_yolo_sam.yaml`. Make sure to provide the correct model verison (vit_h, vit_b) and pretrained model path in the config file.

4. Run the following command to start the OPEN-IRIS with SAM and YOLO model:
```bash
python test_yolo.py --config <path_to_config_file> --image <path_to_image> --mask <path_to_mask> --save_dir <path_to_output> --eye_side <left/right>

# example
python test_yolo.py --config pipeline_yolo_sam.yaml --image data/images/S1001L01.jpg --mask data/masks/S1001L01.png --save_dir output/ --eye_side left
```