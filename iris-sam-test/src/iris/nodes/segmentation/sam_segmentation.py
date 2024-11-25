from __future__ import annotations

import os
from typing import Dict, List, Literal, Tuple, Optional

import numpy as np
import torch
from pydantic import PositiveInt
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.modeling import Sam

from iris.callbacks.callback_interface import Callback
from iris.io.dataclasses import IRImage, SegmentationMap
from iris.nodes.segmentation.multilabel_segmentation_interface import MultilabelSemanticSegmentationInterface


# from segment_anything import  sam_model_registry, SamPredictor
# import torch
import cv2



class SamPredictorMultiClass(SamPredictor):
    def __init__(
        self,
        sam_model: Sam
    ) -> None:
    
        super().__init__(sam_model)
    
    
    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = False, # true not supported
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks for the given input prompts, using the currently set image.
        only multi boxes are supported for multiple objects per image

        Arguments:
          point_coords (np.ndarray or None): A Nx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (np.ndarray or None): A length N array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          box (np.ndarray or None): A length nx4 array given n boxes prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form 1xHxW, where
            for SAM, H=W=256.
        #   multimask_output (bool): If true, the model will return three masks.
        #     For ambiguous input prompts (such as a single click), this will often
        #     produce better masks than a single prediction. If only a single
        #     mask is needed, the model's predicted quality score can be used
        #     to select the best mask. For non-ambiguous prompts, such as multiple
        #     input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (np.ndarray): The output masks in CxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (np.ndarray): An array of length C containing the model's
            predictions for the quality of each mask.
          (np.ndarray): An array of shape CxHxW, where C is the number
            of masks and H=W=256. These low resolution logits can be passed to
            a subsequent iteration as mask input.
        """
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        # Transform input prompts
        coords_torch, labels_torch, box_torch, mask_input_torch = None, None, None, None
        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = self.transform.apply_coords(point_coords, self.original_size)
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
        if box is not None:
            box = self.transform.apply_boxes(box, self.original_size)
            box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)
            box_torch = box_torch[None, :]
        if mask_input is not None:
            mask_input_torch = torch.as_tensor(mask_input, dtype=torch.float, device=self.device)
            mask_input_torch = mask_input_torch[None, :, :, :]
        masks, iou_predictions, low_res_masks = self.predict_torch(
            coords_torch,
            labels_torch,
            box_torch,
            mask_input_torch,
            False,
            return_logits=return_logits,
        )

        masks = masks[0].detach().cpu().numpy()
        iou_predictions = iou_predictions[0].detach().cpu().numpy()
        low_res_masks = low_res_masks[0].detach().cpu().numpy()
        return masks, iou_predictions, low_res_masks
    
    @torch.no_grad()
    def predict_torch(
        self,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = False,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks for the given input prompts, using the currently set image.
        Input prompts are batched torch tensors and are expected to already be
        transformed to the input frame using ResizeLongestSide.

        Arguments:
          point_coords (torch.Tensor or None): A BxNx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (torch.Tensor or None): A BxN array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          boxes (np.ndarray or None): A Bx4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form Bx1xHxW, where
            for SAM, H=W=256. Masks returned by a previous iteration of the
            predict method do not need further transformation.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (torch.Tensor): The output masks in BxCxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (torch.Tensor): An array of shape BxC containing the model's
            predictions for the quality of each mask.
          (torch.Tensor): An array of shape BxCxHxW, where C is the number
            of masks and H=W=256. These low res logits can be passed to
            a subsequent iteration as mask input.
        """
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")
        

        if point_coords is not None:
            points = (point_coords, point_labels)
        else:
            points = None

        # Embed prompts
        image_pe = self.model.prompt_encoder.get_dense_pe()
        low_res_masks_list, iou_predictions_list = [], []
        for i in range(boxes.shape[1]):
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=points,
                boxes=boxes[:, i, :],
                masks=mask_input,
            )
            
            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=self.features,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            low_res_masks_list.append(low_res_masks)
            iou_predictions_list.append(iou_predictions)
        
        low_res_masks = torch.cat(low_res_masks_list, dim=1)
        iou_predictions = torch.cat(iou_predictions_list, dim=1)
        

        # Upscale the masks to the original image resolution
        masks = self.model.postprocess_masks(low_res_masks, self.input_size, self.original_size)

        if not return_logits:
            masks = masks > self.model.mask_threshold

        return masks, iou_predictions, low_res_masks



class SAMSegmentation(MultilabelSemanticSegmentationInterface):
    """Implementation of class which uses ONNX model to perform semantic segmentation maps prediction.

    For more detailed model description check model card available in SEMSEG_MODEL_CARD.md file.
    """

    class Parameters(MultilabelSemanticSegmentationInterface.Parameters):
        """Parameters class for SAMSegmentation objects."""

        predictor: SamPredictor
        device: Literal["cpu", "cuda"]
        # input_resolution: Tuple[PositiveInt, PositiveInt]
        input_num_channels: Literal[1, 3]

    __parameters_type__ = Parameters

    # @classmethod
    # def create_from_hugging_face(
    #     cls,
    #     model_path: str,
    #     model_type: str,
    #     # input_resolution: Tuple[PositiveInt, PositiveInt] = (640, 480),
    #     input_num_channels: Literal[1, 3] = 3,
    #     callbacks: List[Callback] = [],
    # ) -> SAMSegmentation:
    #     """Create SAMSegmentation object with by downloading model from HuggingFace repository `MultilabelSemanticSegmentationInterface.HUGGING_FACE_REPO_ID`.

    #     Args:
    #         model_path (str): path of the pretrained SAM model
    #         model_type (str): type of the model
    #         input_resolution (Tuple[PositiveInt, PositiveInt], optional): Neural Network input image resolution. Defaults to (640, 480).
    #         input_num_channels (Literal[1, 3], optional): Neural Network input image number of channels. Defaults to 3.
    #         callbacks (List[Callback], optional): List of algorithm callbacks. Defaults to [].

    #     Returns:
    #         SAMSegmentation: SAMSegmentation object.
    #     """
    

    #     return SAMSegmentation(model_path, model_type, input_num_channels, callbacks)

    def __init__(
        self,
        model_path: str,
        model_type: str,
        # input_resolution: Tuple[PositiveInt, PositiveInt] = (640, 480),
        input_num_channels: Literal[1, 3] = 3,
        callbacks: List[Callback] = [],
    ) -> None:
        """Assign parameters.

        Args:
            model_path (str): Path to the ONNX model.
            model_type (str): type of the model
            input_resolution (Tuple[PositiveInt, PositiveInt], optional): Neural Network input image resolution. Defaults to (640, 480).
            input_num_channels (Literal[1, 3], optional): Neural Network input image number of channels. Defaults to 3.
            callbacks (List[Callback], optional): List of algorithm callbacks. Defaults to [].
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        sam_model = sam_model_registry[model_type](checkpoint=model_path).to(self.device)
        predictor = SamPredictorMultiClass(sam_model)
        
        # classes mapping for the SAM model
        self.model_classes = {
            'iris': 0,
            'pupil': 1,
            'sclera': 2,
            'top_lashes': 3,
            'bottom_lashes': 4,
        }
        

        super().__init__(
            predictor=predictor,
            device=self.device,
            # input_resolution=input_resolution,
            input_num_channels=input_num_channels,
            callbacks=callbacks,
        )

    def run(self, image: IRImage) -> SegmentationMap:
        """Perform semantic segmentation prediction on an image.

        Args:
            image (IRImage): Infrared image object.

        Returns:
            SegmentationMap: Postprocessed model predictions.
        """
        nn_input = self._preprocess(image.img_data, image.bboxes)

        prediction = self._forward(nn_input)

        return self._postprocess(prediction, original_image_resolution=(image.width, image.height))

    def _preprocess(self, image: np.ndarray, bboxes: np.ndarray) -> Dict[str, np.ndarray]:
        """Preprocess image so that inference with ONNX model is possible.

        Args:
            image (np.ndarray): Infrared image object.

        Returns:
            Dict[str, np.ndarray]: Dictionary with wrapped input name and image data {input_name: image_data}.
        """
        # we dont need to preprocess the image as SAM will take care of it
        
        nn_input = image.copy()
        bboxes_input = bboxes.copy()
        
        # conver gray to RGB
        if nn_input.shape[-1] == 1 or len(nn_input.shape) == 2:
            nn_input = cv2.cvtColor(nn_input, cv2.COLOR_GRAY2RGB)
        
        # use cv2 to get 4 points as a bounding boux from user}
        # bbox = draw_box_points(nn_input, name='abc')
        # print('BBox coordinates provided by user: ', bbox)

        return {'image': nn_input,  'bboxes': bboxes_input}

    def _forward(self, preprocessed_input: Dict[str, np.ndarray]) -> List[np.ndarray]:
        """Neural Network forward pass.

        Args:
            preprocessed_input (Dict[str, np.ndarray]): Input iamge and bbox

        Returns:
            List[np.ndarray]: Predictions.
        """
        self.params.predictor.set_image(preprocessed_input['image'])
        pred_masks, _, _ = self.params.predictor.predict(
            point_coords=None,
            box=preprocessed_input['bboxes'],
            multimask_output=False,
        )
        
        # fix eye masks
        pred_masks = pred_masks.astype(bool)
        pred_masks = self.fix_eye_masks(pred_masks)
        # expand 1 dim to match the expected shape and change to uint8
        pred_masks = np.expand_dims(pred_masks, axis=0).astype(np.uint8)
        
        # save_path ='outputs'
        # os.makedirs(save_path, exist_ok=True)
        # for idx, mask in enumerate(pred_masks[0]):
        #     mask_path = os.path.join(save_path, f'pred_mask_{idx}.png')
        #     mask = np.expand_dims(mask, axis=-1).astype(np.uint8)
        #     print('mask: ', mask.shape, mask.dtype)
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
    
    # def fix_eye_masks(self, masks):
    #     """
    #     a method to post process eye masks based on domain knowledge
        
    #     args:
    #     masks: a tensor or numpy arrays of shape (num_classes, H, W)
    #     returns:
    #     masks: a np array of shape (num_classes, H, W) with post-processed
    #     """
        
    #     if isinstance(masks, torch.Tensor):
    #         masks = masks.cpu().numpy()
        
        
    #     iris_mask = masks[self.model_classes['iris']]
    #     pupil_mask = masks[self.model_classes['pupil']]
    #     sclera_mask = masks[self.model_classes['sclera']]
    #     top_lashes_mask = masks[self.model_classes['top_lashes']]
    #     bottom_lashes_mask = masks[self.model_classes['bottom_lashes']]
        
    #     # expand iris mask to fill holes
    #     iris_mask = cv2.morphologyEx(iris_mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)
    #     # expand pupil mask to fill holes
    #     pupil_mask = cv2.morphologyEx(pupil_mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)
        
    #     # expand sclera mask to fill holes
    #     sclera_mask = cv2.morphologyEx(sclera_mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)
        
        
    #     # flood fill pupil mask
    #     pupil_mask = self.fill_holes(pupil_mask)
    #     pupil_iris_mask = pupil_mask | iris_mask
    #     # floodfill pupil_iris_mask
    #     pupil_iris_mask = self.fill_holes(pupil_iris_mask)
        
    #     # remove pupil mask where it overlaps with iris mask
    #     pupil_mask = pupil_iris_mask & (~iris_mask)
        
    #     # if pupil mask has more than 1 contours, keep the largest
    #     # add the smaller ones to iris mask
    #     contours, _ = cv2.findContours(pupil_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     if len(contours) > 1:
    #         # get the largest contour
    #         contour_areas = [cv2.contourArea(contour) for contour in contours]
    #         largest_idx = np.argmax(contour_areas)
    #         largest_contour = contours[largest_idx]
    #         pupil_mask = cv2.drawContours(np.zeros_like(pupil_mask).astype(np.uint8), [largest_contour], -1, 1, thickness=cv2.FILLED).astype(bool)
    #         # remove pupil mask where it overlaps with iris mask
    #         pupil_mask = pupil_mask & (~iris_mask)
            
    #         # add the smaller contours to iris mask
    #         # remove the largest contour
    #         contours.pop(largest_idx)
    #         for contour in contours:
    #             contour_mask = cv2.drawContours(np.zeros_like(pupil_mask).astype(np.uint8), [contour], -1, 1, thickness=cv2.FILLED).astype(bool)
    #             iris_mask = iris_mask | contour_mask
            
    #     # if iris mask has more than 1 contours, keep the largest
    #     contours, _ = cv2.findContours(iris_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     if len(contours) > 1:
    #         # get the largest contour
    #         contour_areas = [cv2.contourArea(contour) for contour in contours]
    #         largest_idx = np.argmax(contour_areas)
    #         largest_contour = contours[largest_idx]
    #         iris_mask = cv2.drawContours(np.zeros_like(iris_mask).astype(np.uint8), [largest_contour], -1, 1, thickness=cv2.FILLED).astype(bool)
        
    #     pupil_iris_mask = iris_mask | pupil_mask
        
    #     # floodfill sclera_mask
    #     sclera_mask = self.fill_holes(sclera_mask)
    #     eyeball_mask = sclera_mask | pupil_iris_mask
    #     eyeball_mask = self.fill_holes(eyeball_mask)
        
    #     sclera_mask = eyeball_mask & (~pupil_iris_mask)
        
    #     # remove very small contours from sclera mask
    #     contours, _ = cv2.findContours(sclera_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     mask_area = np.sum(sclera_mask)
    #     sclera_mask = np.zeros_like(sclera_mask).astype(bool)
    #     for contour in contours:
    #         # if area is more than 10% of the mask area, keep it
    #         contour_area = cv2.contourArea(contour)
    #         if contour_area < 0.1*mask_area:
    #             continue            
    #         sclera_mask = sclera_mask | cv2.drawContours(np.zeros_like(sclera_mask).astype(np.uint8), [contour], -1, 1, thickness=cv2.FILLED).astype(bool)
        
        
    #     # remove sclera mask where it overlaps with iris mask
    #     # sclera_mask = sclera_mask & (~iris_mask)
        
    #     # merge top and bottom lashes
    #     eyelashes_mask = top_lashes_mask | bottom_lashes_mask
        
    #     # fixed_masks = np.stack([iris_mask, pupil_mask, sclera_mask,  eyelashes_mask], axis=0)
        
    #     # make sure the order matches as of the interface
    #     fixed_masks = np.stack([sclera_mask, iris_mask, pupil_mask, eyelashes_mask], axis=0)
        
    #     return fixed_masks
    
    def fix_eye_masks(self, masks):
        """
        a method to post process eye masks based on domain knowledge
        
        args:
        masks: a tensor or numpy arrays of shape (num_classes, H, W)
        returns:
        masks: a np array of shape (num_classes, H, W) with post-processed
        """
        
        if isinstance(masks, torch.Tensor):
            masks = masks.cpu().numpy()
        
        
        iris_mask = masks[self.model_classes['iris']]
        pupil_mask = masks[self.model_classes['pupil']]
        sclera_mask = masks[self.model_classes['sclera']]
        top_lashes_mask = masks[self.model_classes['top_lashes']]
        bottom_lashes_mask = masks[self.model_classes['bottom_lashes']]
        
        
        # find circle around the pupil
        # Find contours
        contours, _ = cv2.findContours(pupil_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #  we are interested in the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # fill th largest contour as the pupil mask
        pupil_mask = cv2.drawContours(np.zeros_like(pupil_mask).astype(np.uint8), [largest_contour], -1, 1, thickness=cv2.FILLED).astype(bool)

        # # Find the minimum enclosing circle for the largest contour
        # (x, y), radius = cv2.minEnclosingCircle(largest_contour)

        # # Convert the (x, y) to integers
        # pupil_center = (int(x), int(y))
        # pupil_radius = int(radius)

        
        
        # # find the circle around the iris
        # # Find contours
        # contours, _ = cv2.findContours(iris_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # #  we are interested in the largest contour
        # largest_contour = max(contours, key=cv2.contourArea)
    

        # # Find the minimum enclosing circle for the largest contour
        # (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        # iris_center = (int(x), int(y))
        # iris_radius = int(radius)
        
        
        # # fill any gaps in the pupil circle and until it touches iris mask in all directions
        # pupil_mask_filled = np.zeros_like(pupil_mask)
        # cv2.circle(pupil_mask_filled, pupil_center, pupil_radius, 1, thickness=cv2.FILLED).astype(bool)
        # pupil_mask_filled = pupil_mask_filled & iris_mask
        
        pupil_iris_mask = pupil_mask | iris_mask
        
        # floodfill pupil_iris_mask
        # pupil_iris_mask = self.fill_holes(pupil_iris_mask)
        contours, _ = cv2.findContours(pupil_iris_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        pupil_iris_mask = cv2.drawContours(np.zeros_like(pupil_iris_mask).astype(np.uint8), [largest_contour], -1, 1, thickness=cv2.FILLED).astype(bool)
        
        iris_mask = pupil_iris_mask & (~pupil_mask)
        
        eyeball_mask = sclera_mask | pupil_iris_mask
        
        #Find all contours in the mask
        contours, _ = cv2.findContours(eyeball_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Combine all contours into one set of points
        all_points = np.vstack(contours)

        # Find the convex hull that covers all points
        hull = cv2.convexHull(all_points)

        # Draw the convex hull on a new mask
        hull_mask = np.zeros(eyeball_mask.shape, dtype=np.uint8)
        eyeball_mask = cv2.drawContours(hull_mask, [hull], -1, 1, thickness=cv2.FILLED).astype(bool)
        
        
        # sclera_mask = eyeball_mask & (~pupil_iris_mask)
        
        # merge top and bottom lashes
        eyelashes_mask = top_lashes_mask | bottom_lashes_mask
        
        # fixed_masks = np.stack([iris_mask, pupil_mask, sclera_mask,  eyelashes_mask], axis=0)
        
        # make sure the order matches as of the interface
        fixed_masks = np.stack([eyeball_mask, iris_mask, pupil_mask, eyelashes_mask], axis=0)
        
        return fixed_masks
    
    def fill_holes(self, mask):
        # mask to unit8
        mask = mask.astype(np.uint8)
        
        # Invert the mask
        inverted_mask = cv2.bitwise_not(mask)

        # Get the size of the mask
        h, w = mask.shape[:2]

        # Create a mask for floodFill
        flood_fill_mask = np.zeros((h + 2, w + 2), np.uint8)

        # Flood fill from point (0, 0)
        cv2.floodFill(inverted_mask, flood_fill_mask, (0, 0), 255)

        # Invert the flood-filled image
        flood_filled_inverted = cv2.bitwise_not(inverted_mask)

        # Combine the original mask and the flood-filled image
        filled_mask = cv2.bitwise_or(mask, flood_filled_inverted)

        return filled_mask.astype(bool)