try:
    from iris.nodes.segmentation import tensorrt_multilabel_segmentation

    MultilabelSegmentation = tensorrt_multilabel_segmentation.TensorRTMultilabelSegmentation
except ModuleNotFoundError:
    from iris.nodes.segmentation import onnx_multilabel_segmentation

    MultilabelSegmentation = onnx_multilabel_segmentation.ONNXMultilabelSegmentation
