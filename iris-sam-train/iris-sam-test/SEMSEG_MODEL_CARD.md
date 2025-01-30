# Iris Semantic Segmentation Model Card

## Model Overview

The content on this card pertains to a model that conducts semantic segmentation of the eye. When provided with an infrared image, the model assigns labels to pixels based on the various classes it can differentiate: `eyeball`, `iris`, `pupil`, and `eyelashes`.

## Model Architecture

The model architecture is based on the UNet++ [1]. UNet++ represents a complete convolutional neural network designed for semantic segmentation of images. It comprises both an encoder and a decoder, linked through skip connections. The encoder captures features with varying spatial resolutions (utilizing skip connections), which the decoder leverages to create precise segmentation masks. Notably, the decoder within Unet++ possesses greater complexity compared to the standard Unet model.

The MobileNetV2 [2] architecture serves as the encoder for our model. During the inference phase, the encoder produces five sets of feature map tensors, each corresponding to distinct depths of encoding. Subsequent feature map tensors exhibit a reduction of spatial dimensions by a factor of two compared to their preceding counterparts. For instance, a tensor generated at depth 0 will possess dimensions [(batch_size, num_channels, height, width)], while a tensor generated at depth 1 will exhibit dimensions [(batch_size, num_channels, height / 2, width / 2)].

The decoder is enhanced through the incorporation of Spatial and Channel "Squeeze & Excitation" (scSE) blocks [3], which are appended to each decoding layer.

## Dataset

Importantly, no Worldcoin user data was used to train or fine-tune the IRIS pipeline. Rather, a research datasets were used to train model. Datasets used during training comes from:
 - ND-IRIS-0405 [4]
 - CASIA-Iris-Interval [5]
 - CASIA-Iris-Lamp [5]
 - CASIA-Iris-Thousand [5]
 - CASIA-Iris-Twins [5]
 - IIT Delhi Iris Database (IITD) [6]
 - LivDet2013-Iris [7]
 - LivDet2015-Iris [8]
 - LivDet2017-Iris [9]
 - Multimedia University Iris Database (MMU) [10]

The experimental dataset contained a total of 18 431 annotated IR images. Table below presents dataset split used during training semantic segmentation model.


| **Dataset type**| **Number of images** |
|-----------------------------|-----------|
| train               | 14685     |
| validation               | 1880     |
| test               | 1866     |

## Performance

### Semantic Segmentation Performance

#### Measured on entire dataset

| **Metric**                  | **Value** |
|-----------------------------|-----------|
| _eyeball_IoU_               | 0.986     |
| _iris_IoU_                  | 0.978     |
| _pupil_IoU_                 | 0.978     |
| _eyelashes_IoU_             | 0.798     |
| _mIoU_                      | 0.943     |
| _eyelashes_inside_iris_IoU_ | 0.791     |
| _boundary_confidence_pupil_ | 0.965     |
| _boundary_confidence_iris_  | 0.907     |
| _chamfer_eyeball_           | 1.689     |
| _chamfer_iris_              | 1.868     |
| _chamfer_pupil_             | 0.680     |

#### Measured on particular dataset

|Name                                                           |test_IoU_eyeball|test_IoU_eyelashes|test_IoU_iris|test_IoU_pupil|test_boundary_confidence_iris|test_boundary_confidence_pupil|test_chamfer_dist_eyeball|test_chamfer_dist_iris|test_chamfer_dist_pupil|test_mIoU |test_mIoU_eyelashes_inside_iris|
|---------------------------------------------------------------|------------------------|--------------------------|---------------------|----------------------|-----------------------------|------------------------------|---------------------------------|------------------------------|-------------------------------|------------------|---------------------------------------|
|CASIA-Iris-Interval                                            |0.992                   |0.885                     |0.977                |0.983                 |0.868                        |0.959                         |1.230                            |3.132                         |0.673                          |0.970             |0.876                                  |
|CASIA-Iris-Lamp                                                |0.987                   |0.844                     |0.988                |0.983                 |0.979                        |0.988                         |0.683                            |0.482                         |0.295                          |0.953             |0.836                                  |
|CASIA-Iris-Thousand                                            |0.987                   |0.823                     |0.989                |0.981                 |0.981                        |0.988                         |0.637                            |0.430                         |0.252                          |0.956             |0.767                                  |
|CASIA-Iris-Twins                                               |0.986                   |0.837                     |0.986                |0.981                 |0.966                        |0.984                         |0.998                            |0.865                         |0.710                          |0.949             |0.840                                  |
|IITD                                                           |0.990                   |0.868                     |0.975                |0.979                 |0.843                        |0.944                         |3.424                            |3.564                         |0.939                          |0.966             |0.861                                  |
|LivDet2013-Iris                                                |0.980                   |0.853                     |0.985                |0.977                 |0.973                        |0.977                         |1.242                            |0.518                         |0.393                          |0.945             |0.851                                  |
|LivDet2015-Iris                                                |0.978                   |0.844                     |0.984                |0.974                 |0.955                        |0.958                         |1.524                            |0.794                         |0.608                          |0.938             |0.870                                  |
|LivDet2017-Iris                                                |0.978                   |0.791                     |0.981                |0.956                 |0.956                        |0.973                         |1.920                            |1.274                         |1.204                          |0.921             |0.738                                  |
|MMU                                                            |0.983                   |0.845                     |0.985                |0.980                 |0.960                        |0.980                         |1.405                            |0.821                         |0.500                          |0.946             |0.765                                  |
|ND-IRIS-0405                                                   |0.981                   |0.750                     |0.977                |0.975                 |0.918                        |0.969                         |1.757                            |1.653                         |0.739                          |0.917             |0.676                                  |


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

### Example inference results

**Note**: The provided input image has been subjected to the processing methodology described earlier, prior to its introduction into the model. Moreover, for the intent of visualization, the IR image presented has been anonymized to safeguard the identity of the user. It is also worth to note that the inference process was conducted on the original, non-anonymized version of the image.

#### Input image

<img src="https://github.com/worldcoin/open-iris/blob/main/docs/model_card/anonymized.png" alt="anonymized input image" width=640 height=480>

#### Output masks

**Note**: A threshold of 0.5 was applied on every resultant probability matrix to generate a definitive binary mask for each individual class. Also, every outputted segmentation map was resized to the input image resolution.

<img src="https://github.com/worldcoin/open-iris/blob/main/docs/model_card/overlayed_segmaps.png" alt="segmaps">

## Limitations

Thorough examination of the results enabled us to pinpoint situations where the segmentation model experiences declines in performance. These instances are as follows:
- Segmenting images with high specular reflection coming usually from glasses may lead to bad segmentation map predictions.
- Data based on which the model was trained were captured in the constrained environment with cooperative users. Therefore, in practice model is expected to produce poor segmentation maps for cases like: offgaze, misaligned eyes, blurry images etc.

## Further reading

1. Zhou, Z., Rahman Siddiquee, M. M., Tajbakhsh, N., & Liang, J. (2018). UNet++: A nested U-Net Architecture for Medical Image Segmentation. (https://arxiv.org/abs/1807.10165v1)
2. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. (https://arxiv.org/abs/1801.04381)
3. Roy, A. G., Navab, N., & Wachinger, C. (2018). Recalibrating Fully Convolutional Networks with Spatial and Channel “Squeeze and Excitation” Blocks. (https://arxiv.org/abs/1808.08127v1)
4. Bowyer, K. , Flynn, P. (2016), The ND-IRIS-0405 Iris Image Dataset (https://arxiv.org/abs/1606.04853)
5. http://biometrics.idealtest.org/
6. http://www4.comp.polyu.edu.hk/csajaykr/IITD/Database_Iris.htm
7. D. Yambay, J.S. Doyle, K.W. Bowyer, A. Czajka, S. Schuckers Livdet-iris 2013 - iris liveness detection competition 2013
8. D. Yambay, B. Walczak, S. Schuckers, A. Czajka Livdet-iris 2015 - iris liveness detection competition 2015
9. D. Yambay, B. Becker, N. Kohli, D. Yadav, A. Czajka, K.W. Bowyer, S. Schuckers, R. Singh, M.Vatsa, A. Noore, D. Gragnaniello, C. Sansone, L. Verdoliva, L. He, Y. Ru, H. Li, N. Liu, Z. Sun, T. Tan Livdet iris 2017 - iris liveness detection competition 2017
10. https://mmuexpert.mmu.edu.my/ccteo
