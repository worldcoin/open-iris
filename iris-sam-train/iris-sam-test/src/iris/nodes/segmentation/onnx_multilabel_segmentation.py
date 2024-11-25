from __future__ import annotations

import os
from typing import Dict, List, Literal, Tuple

import numpy as np
import onnx
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from pydantic import PositiveInt

from iris.callbacks.callback_interface import Callback
from iris.io.dataclasses import IRImage, SegmentationMap
from iris.nodes.segmentation.multilabel_segmentation_interface import MultilabelSemanticSegmentationInterface


class ONNXMultilabelSegmentation(MultilabelSemanticSegmentationInterface):
    """Implementation of class which uses ONNX model to perform semantic segmentation maps prediction.

    For more detailed model description check model card available in SEMSEG_MODEL_CARD.md file.
    """

    class Parameters(MultilabelSemanticSegmentationInterface.Parameters):
        """Parameters class for ONNXMultilabelSegmentation objects."""

        session: ort.InferenceSession
        input_resolution: Tuple[PositiveInt, PositiveInt]
        input_num_channels: Literal[1, 3]

    __parameters_type__ = Parameters

    @classmethod
    def create_from_hugging_face(
        cls,
        model_name: str = "iris_semseg_upp_scse_mobilenetv2.onnx",
        input_resolution: Tuple[PositiveInt, PositiveInt] = (640, 480),
        input_num_channels: Literal[1, 3] = 3,
        callbacks: List[Callback] = [],
    ) -> ONNXMultilabelSegmentation:
        """Create ONNXMultilabelSegmentation object with by downloading model from HuggingFace repository `MultilabelSemanticSegmentationInterface.HUGGING_FACE_REPO_ID`.

        Args:
            model_name (str, optional): Name of the ONNX model stored in HuggingFace repo. Defaults to "iris_semseg_upp_scse_mobilenetv2.onnx".
            input_resolution (Tuple[PositiveInt, PositiveInt], optional): Neural Network input image resolution. Defaults to (640, 480).
            input_num_channels (Literal[1, 3], optional): Neural Network input image number of channels. Defaults to 3.
            callbacks (List[Callback], optional): List of algorithm callbacks. Defaults to [].

        Returns:
            ONNXMultilabelSegmentation: ONNXMultilabelSegmentation object.
        """
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        model_path = hf_hub_download(
            repo_id=MultilabelSemanticSegmentationInterface.HUGGING_FACE_REPO_ID,
            cache_dir=MultilabelSemanticSegmentationInterface.MODEL_CACHE_DIR,
            filename=model_name,
        )

        return ONNXMultilabelSegmentation(model_path, input_resolution, input_num_channels, callbacks)

    def __init__(
        self,
        model_path: str,
        input_resolution: Tuple[PositiveInt, PositiveInt] = (640, 480),
        input_num_channels: Literal[1, 3] = 3,
        callbacks: List[Callback] = [],
    ) -> None:
        """Assign parameters.

        Args:
            model_path (str): Path to the ONNX model.
            input_resolution (Tuple[PositiveInt, PositiveInt], optional): Neural Network input image resolution. Defaults to (640, 480).
            input_num_channels (Literal[1, 3], optional): Neural Network input image number of channels. Defaults to 3.
            callbacks (List[Callback], optional): List of algorithm callbacks. Defaults to [].
        """
        onnx_model = onnx.load(model_path)
        onnx.checker.check_model(onnx_model)

        super().__init__(
            session=ort.InferenceSession(model_path, providers=["CPUExecutionProvider"]),
            input_resolution=input_resolution,
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
        nn_input = image.copy()

        nn_input = self.preprocess(nn_input, self.params.input_resolution, self.params.input_num_channels)

        return {self.params.session.get_inputs()[0].name: nn_input.astype(np.float32)}

    def _forward(self, preprocessed_input: Dict[str, np.ndarray]) -> List[np.ndarray]:
        """Neural Network forward pass.

        Args:
            preprocessed_input (Dict[str, np.ndarray]): Inputs.

        Returns:
            List[np.ndarray]: Predictions.
        """
        return self.params.session.run(None, preprocessed_input)

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
