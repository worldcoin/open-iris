try:
    from iris.nodes.segmentation import tensorrt_multilabel_segmentation

    MultilabelSegmentation = tensorrt_multilabel_segmentation.TensorRTMultilabelSegmentation
except ModuleNotFoundError:
    from iris.nodes.segmentation import onnx_multilabel_segmentation

    MultilabelSegmentation = onnx_multilabel_segmentation.ONNXMultilabelSegmentation
from iris.nodes.segmentation import sam_segmentation, yolo_sam_segmentation
SAMSegmentation = sam_segmentation.SAMSegmentation
YOLOSAMSegmentation = yolo_sam_segmentation.YOLOSAMSegmentation