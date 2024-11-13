from __future__ import annotations

import os
from typing import Dict, List, Literal, Tuple, Optional

import numpy as np
import torch
from ultralytics import YOLO

from iris.callbacks.callback_interface import Callback
from iris.io.dataclasses import IRImage, SegmentationMap
from iris.nodes.segmentation.multilabel_segmentation_interface import MultilabelSemanticSegmentationInterface

from iris.nodes.segmentation.sam_segmentation import SamPredictor, SAMSegmentation

# import torch
import cv2


class YOLOSAMSegmentation(SAMSegmentation):
    """Implementation of class which uses ONNX model to perform semantic segmentation maps prediction.

    For more detailed model description check model card available in SEMSEG_MODEL_CARD.md file.
    
    Thi uses YOLO model to detect part of the eyes and then uses SAM model to segment the eyes into different classes.
    """

    class Parameters(MultilabelSemanticSegmentationInterface.Parameters):
        """Parameters class for SAMSegmentation objects."""

        predictor: SamPredictor
        device: Literal["cpu", "cuda"]
        # input_resolution: Tuple[PositiveInt, PositiveInt]
        input_num_channels: Literal[1, 3]

    __parameters_type__ = Parameters


    def __init__(
        self,
        model_path: str, # sam model path
        model_type: str, # sam model type
        yolo_model_path: str, # yolo model path
        # input_resolution: Tuple[PositiveInt, PositiveInt] = (640, 480),
        input_num_channels: Literal[1, 3] = 3,
        conf_thresh: float = 0.2,
        iou_thresh: float = 0.7,
        agnostic: bool = False,
        max_det: int = 10,
        callbacks: List[Callback] = [],
    ) -> None:
        """Assign parameters.

        Args:
            model_path (str): Path to the ONNX model.
            model_type (str): type of the model
            input_resolution (Tuple[PositiveInt, PositiveInt], optional): Neural Network input image resolution. Defaults to (640, 480).
            input_num_channels (Literal[1, 3], optional): Neural Network input image number of channels. Defaults to 3.
            conf_thresh (float, optional): minimal accepted confidence for object filtering. Defaults to 0.2.
            iou_thresh (float, optional): minimal overlap score for removing objects duplicates in NMS. Defaults to 0.7.
            agnostic (bool, optional): apply class agnostinc NMS approach or not. Defaults to False.
            max_det (int, optional): maximum detections after NMS. Defaults to 10.
            callbacks (List[Callback], optional): List of algorithm callbacks. Defaults to [].
        """
        
        super().__init__(model_path, model_type, input_num_channels, callbacks)
        
        # load the yolo model
        self.yolo_model = YOLO(yolo_model_path).to(self.device)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.agnostic = agnostic
        self.max_det = max_det
        # self.count = 0
    

    def run(self, image: IRImage) -> SegmentationMap:
        """Perform semantic segmentation prediction on an image.

        Args:
            image (IRImage): Infrared image object.

        Returns:
            SegmentationMap: Postprocessed model predictions.
        """
        nn_input = self._preprocess(image.img_data)

        prediction = self._forward(nn_input)

        return self._postprocess(prediction, original_image_resolution=(image.width, image.height))

    def _preprocess(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Preprocess image so that inference with ONNX model is possible.

        Args:
            image (np.ndarray): Infrared image object.

        Returns:
            Dict[str, np.ndarray]: Dictionary with wrapped input name and image data {input_name: image_data}.
        """
        # we dont need to preprocess the image as SAM will take care of it
        
        nn_input = image.copy()
        
        # conver gray to RGB
        if nn_input.shape[-1] == 1 or len(nn_input.shape) == 2:
            nn_input = cv2.cvtColor(nn_input, cv2.COLOR_GRAY2RGB)
        
        return {'image': nn_input}

    def _forward(self, preprocessed_input: Dict[str, np.ndarray]) -> List[np.ndarray]:
        """Neural Network forward pass.

        Args:
            preprocessed_input (Dict[str, np.ndarray]): Input iamge and bbox

        Returns:
            List[np.ndarray]: Predictions.
        """
        # get the detections from yolo model
        detections = self.predict_yolo([preprocessed_input['image']],
                                       conf_thres=self.conf_thresh,
                                        iou_thresh=self.iou_thresh,
                                        agnostic=self.agnostic,
                                        max_det=self.max_det)[0]
        
        # remove multiple objects of same calss with low confidence
        # sort by confidence in descending order
        sorted_indices = np.argsort(-detections[:, 4]) 
        detections = detections[sorted_indices]
        _, unique_indices = np.unique(detections[:, 5], return_index=True)
        detections = detections[unique_indices]
        
        # reorder the detections by class ids in the last column
        detections = detections[detections[:, -1].argsort()]
        
        missing_classes = set(range(1, 5)) - set(detections[:, 5]) # no object detected for these classes
        bboxes = detections[:, :4]
        
        self.params.predictor.set_image(preprocessed_input['image'])
        pred_masks, _, _ = self.params.predictor.predict(
            point_coords=None,
            box=bboxes,
            multimask_output=False,
        )
        
        # add empty masks for missing classes
        for cls in sorted(missing_classes):
            pred_masks = np.insert(pred_masks, cls, np.zeros_like(pred_masks[0]), axis=0)
        
        # fix eye masks
        pred_masks = pred_masks.astype(bool)
        
        pred_masks = self.fix_eye_masks(pred_masks)
        # expand 1 dim to match the expected shape and change to uint8
        pred_masks = np.expand_dims(pred_masks, axis=0).astype(np.uint8)
        # self.count += 1
        # save_path ='outputs/testing'
        # os.makedirs(save_path, exist_ok=True)
        # for idx, mask in enumerate(pred_masks[0]):
        #     mask_path = os.path.join(save_path, f'{self.count}_pred_mask_{idx}.png')
        #     mask = np.expand_dims(mask, axis=-1).astype(np.uint8)
        #     cv2.imwrite(mask_path, mask*255)
            
        return [pred_masks]

    def _postprocess(self, nn_output: List[np.ndarray], original_image_resolution: Tuple[int, int]) -> SegmentationMap:
        """Postprocess model prediction and wrap it within SegmentationMap object for further processing.

        Args:
            nn_output (List[np.ndarray]): Neural Network output. Should be of length equal to 2.
            original_image_resolution (Tuple[int, int]): Original image resolution used to resize predicted semantic segmentation maps.

        Returns:
            SegmentationMap: Postprocessed model predictions.
        """
        segmaps_tensor = nn_output[0]

        segmaps_tensor = self.postprocess_segmap(segmaps_tensor, original_image_resolution)

        segmap = SegmentationMap(
            predictions=segmaps_tensor, index2class=MultilabelSemanticSegmentationInterface.CLASSES_MAPPING
        )

        return segmap
    
    def predict_yolo(self, images, conf_thres=0.2, iou_thresh=0.7, agnostic=False, max_det=10, batch_size=1):
        """given a yolov8 model and a list of images, predicts the detections

        Args:
            images: list of images
            conf_thres (float, *optional*, 0.3): minimal accepted confidence for object filtering
            iou_thres (float, *optional*, 0.7): minimal overlap score for removing objects duplicates in NMS
            agnostic (bool, *optiona*, False): apply class agnostinc NMS approach or not
            max_det (int, *optional*, 10):  maximum detections after NMS
            batch_size (int *optional*, 16): Batch size for prediction in case of multiple images
        """
        detections = []
        
        assert batch_size > 0, 'Batch size msut be greater than 0.'
        batch_size = batch_size if len(images) > batch_size else len(images)
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            if not len(batch):
                continue
            preds = self.yolo_model.predict(batch, conf=conf_thres, iou=iou_thresh,
                                agnostic_nms=agnostic, max_det=max_det, verbose=False)
            for pred in preds:
                boxes = pred.boxes#.cpu().numpy()
                boxes, conf, cls = boxes.xyxy, boxes.conf, boxes.cls
                boxes = torch.cat([boxes, conf.unsqueeze(-1), cls.unsqueeze(-1)], dim=1)
                
                detections.append(boxes.cpu().numpy())

        
        return detections
    