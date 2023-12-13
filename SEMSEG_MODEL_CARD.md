# Iris Semantic Segmentation Model Card

## Model Overview

The content on this card pertains to a model that conducts semantic segmentation of the eye. When provided with an infrared image, the model assigns labels to pixels based on the various classes it can differentiate: `eyeball`, `iris`, `pupil`, and `eyelashes`.

## Model Architecture

The model architecture is based on the Unet++. Unet++ represents a complete convolutional neural network designed for semantic segmentation of images. It comprises both an encoder and a decoder, linked through skip connections. The encoder captures features with varying spatial resolutions (utilizing skip connections), which the decoder leverages to create precise segmentation masks. Notably, the decoder within Unet++ possesses greater complexity compared to the standard Unet model.

The MobileNetV2 architecture serves as the encoder for our model. During the inference phase, the encoder produces five sets of feature map tensors, each corresponding to distinct depths of encoding. Subsequent feature map tensors exhibit a reduction of spatial dimensions by a factor of two compared to their preceding counterparts. For instance, a tensor generated at depth 0 will possess dimensions [(batch_size, num_channels, height, width)], while a tensor generated at depth 1 will exhibit dimensions [(batch_size, num_channels, height / 2, width / 2)].

The decoder is enhanced through the incorporation of Spatial and Channel "Squeeze & Excitation" (scSE) blocks, which are appended to each decoding layer.

## Dataset

The experimental dataset contained a total of 9 957 manually annotated IR images comming from 676 different people. All images were captured using LG 4000 device. Table below presents dataset split used during training semantic segmentation model.

| **Dataset type**| **Number of images** | **Number of subject** |
|-----------------------------|-----------|-----------|
| train               | 7933     | 541     |
| validation               | 1032     | 67     |
| test               | 1030     | 68     |

## Performance

### Semantic Segmentation Performance

#### Measured on entire dataset

| **Metric**                  | **Value** |
|-----------------------------|-----------|
| _eyeball_IoU_               | 0.981     |
| _iris_IoU_                  | 0.977     |
| _pupil_IoU_                 | 0.976     |
| _eyelashes_IoU_             | 0.755     |
| _mIoU_                      | 0.918     |
| _eyelashes_inside_iris_IoU_ | 0.683     |
| _boundary_confidence_pupil_ | 0.971     |
| _boundary_confidence_iris_  | 0.921     |
| _chamfer_eyeball_           | 1.667     |
| _chamfer_iris_              | 1.632     |
| _chamfer_pupil_             | 0.686     |

### Time Performance

#### Local machine

| Model | PyTorch CPU      | PyTorch GPU       | ONNX CPU        | ONNX GPU          |
|-------|------------------|-------------------|------------------|------------------|
| Time (ms)  | 295  ± 4.52  | 12.5  ± 0.42  | 193  ± 0  | 12.2  ± 0.75  |

#### Orb machine

| Model | TensorRT 32-bits      | TensorRT 16-bits       |
|-------|------------------|-------------------|
| Time (ms)  | 115  ± 50.9  | 53.84  ± 7.8  |

## How to use the model

### Input

The model expects an infrared image wrapped into a tensor with dimensions of Nx3x640x480 (representing batch size, number of channels, height, and width). Image pixels must be scaled to 1 / 255.0 and subjected to normalization using the z-score normalization algorithm. The normalization process employs mean values of [0.485, 0.456, 0.406] and standard deviations of [0.229, 0.224, 0.225] for each respective channel. It is important to acknowledge that although IR images are inherently grayscale, the inclusion of three channels might appear perplexing upon initial observation. This peculiarity arises from the incorporation of Transfer Learning techniques during training, wherein a MobileNetV2 model pre-trained on the RGB-rich ImageNet dataset was employed.

### Output


The model yields a tensor characterized by dimensions of Nx4x640x480, denoting batch size, segmented class, height, and width, respectively. The correspondence between the tensor's segmented class index and the associated class designation is delineated as follows:
 - 0: `eyeball`
 - 1: `iris`
 - 2: `pupil`
 - 3: `eyelashes`

Within the ambit of each class, the model formulates probability estimates pertaining to the likelihood of a given pixel being attributed to a particular class.

### Examplary inference results

**Note**: The provided input image has been subjected to the processing methodology described earlier, prior to its introduction into the model. Moreover, for the intent of visualization, the IR image presented has been anonymized to safeguard the identity of the user. It is also worth to note that the inference process was conducted on the original, non-anonymized version of the image.

#### Input image

<img src="https://github.com/worldcoin/open-iris/blob/main/docs/model_card/anonymized.png" alt="anonymized input image" width=640 height=480>

#### Output masks

**Note**: A threshold of 0.5 was applied on every resultant probability matrix to generate a definitive binary mask for each individual class. Also, every outputted segmentation map was resized to the input image resolution.

<img src="https://github.com/worldcoin/open-iris/blob/main/docs/model_card/overlayed_segmaps.png" alt="segmaps">

## Limitations

Thorough examination of the results enabled us to pinpoint situations where the segmentation model experiences declines in performance. These instances are as follows:
- Segmenting images that were not captured by LG4400 sensor may not always produce smooth segmentation maps. The segmention performance depends on how similar images to be segmented are to the images captured by LG4400 sensor.
- Segmenting images with high specular reflection comming usually from glasses may lead to bad segmentation map predictions.
- Data based on which the model was trained were captured in the constrained environment with cooperative users. Therefore, in practise model is expected to produce poor segmentation maps for cases like: offgazes, misaligned eyes, blurry images etc.

## Further reading
- [UNet++ paper](https://arxiv.org/abs/1807.10165v1)
- [MobileNetV2 paper](https://arxiv.org/abs/1801.04381)
- [scSE attention paper](https://arxiv.org/abs/1808.08127v1)
