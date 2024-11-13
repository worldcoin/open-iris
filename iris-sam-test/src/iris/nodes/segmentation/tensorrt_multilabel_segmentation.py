from __future__ import annotations

import os
from typing import List, Literal, Tuple

import numpy as np
import pycuda
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
from huggingface_hub import hf_hub_download

from iris.callbacks.callback_interface import Callback
from iris.io.dataclasses import IRImage, SegmentationMap
from iris.nodes.segmentation.multilabel_segmentation_interface import MultilabelSemanticSegmentationInterface


class HostDeviceMem:
    """Class representing host memory."""

    def __init__(self, host_mem: np.ndarray, device_mem: pycuda._driver.DeviceAllocation) -> None:
        """Assign parameters.

        Args:
            host_mem (np.ndarray): Host memory.
            device_mem (pycuda._driver.DeviceAllocation): Allocation device.
        """
        self.host = host_mem
        self.device = device_mem

    def __str__(self) -> str:
        """Create str representation of a class.

        Returns:
            str: String representation of the object.
        """
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self) -> str:
        """Represent class as a str.

        Returns:
            str: String representation of the object.
        """
        return self.__str__()


class TensorRTMultilabelSegmentation(MultilabelSemanticSegmentationInterface):
    """TensorRTTwoHeadedModelSegmentation class implements full interference pipeline for eye segmentation using TensorRT model.

    For more detailed model description check model card available in SEMSEG_MODEL_CARD.md.
    """

    class Parameters(MultilabelSemanticSegmentationInterface.Parameters):
        """Parameter class for TensorRTMultilabelSegmentation class."""

        engine: trt.tensorrt.ICudaEngine
        input_num_channels: Literal[1, 3]
        segmap_output_shape: trt.tensorrt.Dims
        inputs: List[HostDeviceMem]
        outputs: List[HostDeviceMem]
        bindings: List[int]
        stream: pycuda._driver.Stream
        context: trt.tensorrt.IExecutionContext
        pagelocked_buffer: np.ndarray

    __parameters_type__ = Parameters

    @classmethod
    def create_from_hugging_face(
        cls,
        model_name: str = "iris_semseg_upp_scse_mobilenetv2.engine",
        input_num_channels: Literal[1, 3] = 3,
        callbacks: List[Callback] = [],
    ) -> TensorRTMultilabelSegmentation:
        """Create TensorRTMultilabelSegmentation object with by downloading model from HuggingFace repository `MultilabelSemanticSegmentationInterface.HUGGING_FACE_REPO_ID`.

        Args:
            model_name (str, optional): Name of the ONNX model stored in HuggingFace repo. Defaults to "iris_semseg_upp_scse_mobilenetv2.engine".
            input_num_channels (Literal[1, 3]): Model input image number of channels. Defaults to 3.
            callbacks (List[Callback], optional): List of algorithm callbacks. Defaults to [].

        Returns:
            TensorRTMultilabelSegmentation: TensorRTMultilabelSegmentation object.
        """
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        model_path = hf_hub_download(
            repo_id=MultilabelSemanticSegmentationInterface.HUGGING_FACE_REPO_ID,
            cache_dir=MultilabelSemanticSegmentationInterface.MODEL_CACHE_DIR,
            filename=model_name,
        )

        return TensorRTMultilabelSegmentation(model_path, input_num_channels, callbacks)

    def __init__(
        self,
        model_path: str,
        input_num_channels: Literal[1, 3] = 3,
        callbacks: List[Callback] = [],
    ) -> None:
        """Assign parameters.

        Args:
            model_path (str): Path to the TensorRT model.
            input_num_channels (Literal[1, 3]): Model input image number of channels. Defaults to 3.
            callbacks (List[Callback], optional): List of algorithm callbacks. Defaults to [].
        """
        engine = self._load_engine(model_path)

        segmap_output_shape = engine.get_binding_shape(1)
        inputs, outputs, bindings, stream = self._allocate_buffers(engine)
        context = engine.create_execution_context()
        pagelocked_buffer = inputs[0].host

        super().__init__(
            engine=engine,
            input_num_channels=input_num_channels,
            segmap_output_shape=segmap_output_shape,
            inputs=inputs,
            outputs=outputs,
            bindings=bindings,
            stream=stream,
            context=context,
            pagelocked_buffer=pagelocked_buffer,
            callbacks=callbacks,
        )

    def run(self, image: IRImage) -> SegmentationMap:
        """Predicts segmentation maps.

        Args:
            image (IRImage): Image object.

        Returns:
            SegmentationMap: Segmentation maps.
        """
        preprocessed_image = self._preprocess(image.img_data)

        np.copyto(self.params.pagelocked_buffer, preprocessed_image.ravel())
        predictions = self._run_engine(
            self.params.context, self.params.bindings, self.params.inputs, self.params.outputs, self.params.stream
        )

        image_shape = (image.width, image.height)
        predictions = self._postprocess(predictions, image_shape)

        return SegmentationMap(
            predictions=predictions, index2class=MultilabelSemanticSegmentationInterface.CLASSES_MAPPING
        )

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess only image for inference.

        Args:
            image (np.ndarray): Image.

        Returns:
            np.ndarray: Preprocessed image.
        """
        input_height, input_width = self.params.engine.get_binding_shape(0)[2:4]

        nn_input = self.preprocess(image, (input_width, input_height), self.params.input_num_channels)

        return nn_input

    def _postprocess(
        self,
        predictions: List[np.ndarray],
        original_image_size: Tuple[int, int],
    ) -> np.ndarray:
        """Postprocessed model output.

        Args:
            predictions (List[np.ndarray]]): Model output.
            original_image_size (Tuple[int, int]): Original image size (width, height), used to upsample image to original image size.

        Returns:
            np.ndarray: Postprocessed model output.
        """
        segmaps_tensor = predictions[0].reshape(self.params.segmap_output_shape)
        segmaps_tensor = self.postprocess_segmap(segmaps_tensor, original_image_size)

        return segmaps_tensor

    def _load_engine(self, path: str) -> trt.tensorrt.ICudaEngine:
        """Load a trt engine.

        Args:
            path (str): Path to engine.

        Raises:
            RuntimeError: When loading the engine process fails.

        Returns:
            trt.tensorrt.ICudaEngine: Model engine.
        """
        trt_logger = trt.Logger(trt.Logger.WARNING)

        with open(path, "rb") as f, trt.Runtime(trt_logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())

        if engine is None:
            raise RuntimeError(f"Engine evaluates to None! File {path} seems to be broken!")

        return engine

    def _allocate_buffers(self, engine: trt.tensorrt.ICudaEngine) -> Tuple[list, list, list, pycuda._driver.Stream]:
        """Allocates all buffers needed to perform inference.

        Args:
            trt.tensorrt.ICudaEngine: CUDA engine.

        Returns:
            Tuple[list, list, list, pycuda._driver.Stream]: Tuple with needed buffers.
        """
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding))
            engine_dtype = engine.get_binding_dtype(binding)
            dtype = trt.nptype(engine_dtype)

            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))

            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))

        return inputs, outputs, bindings, stream

    def _run_engine(
        self,
        context: trt.tensorrt.IExecutionContext,
        bindings: List[int],
        inputs: List[HostDeviceMem],
        outputs: List[HostDeviceMem],
        stream: pycuda._driver.Stream,
    ) -> List[np.ndarray]:
        """Run the engine and makes inference.

        Args:
            context (trt.tensorrt.IExecutionContext): Execution context.
            bindings (List[int]): Memory buffer bindings.
            inputs (List[HostDeviceMem]): Inputs buffers.
            outputs (List[HostDeviceMem]): Outputs buffers.
            stream (pycuda._driver.Stream): CUDA stream.

        Returns:
            List[np.ndarray]: Predictions.
        """
        # Transfer input data to GPU
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]

        # Run inference
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

        # Transfer predictions back from GPU
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]

        # Synchronize the stream
        stream.synchronize()

        return [out.host for out in outputs]
