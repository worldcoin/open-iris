from __future__ import annotations

from functools import cached_property
from typing import Any, Callable, Dict, List, Literal, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field, NonNegativeInt, root_validator, validator

from iris.callbacks.pipeline_trace import PipelineCallTraceStorage
from iris.io import validators as v
from iris.io.class_configs import ImmutableModel
from iris.utils.base64_encoding import (
    base64_decode_array,
    base64_decode_float_array,
    base64_encode_array,
    base64_encode_float_array,
)
from iris.utils.math import estimate_diameter


class IRImage(ImmutableModel):
    """Data holder for input IR image."""

    img_data: np.ndarray
    eye_side: Literal["left", "right"]

    @property
    def height(self) -> int:
        """Return IR image's height.

        Return:
            int: image height.
        """
        return self.img_data.shape[0]

    @property
    def width(self) -> int:
        """Return IR image's width.

        Return:
            int: image width.
        """
        return self.img_data.shape[1]

    def serialize(self) -> Dict[str, Any]:
        """Serialize IRImage object.

        Returns:
            Dict[str, Any]: Serialized object.
        """
        return self.dict(by_alias=True)

    @staticmethod
    def deserialize(data: Dict[str, Any]) -> IRImage:
        """Deserialize IRImage object.

        Args:
            data (Dict[str, Any]): Serialized object to dict.

        Returns:
            IRImage: Deserialized object.
        """
        return IRImage(**data)


class SegmentationMap(ImmutableModel):
    """Data holder for the segmentation models predictions."""

    predictions: np.ndarray
    index2class: Dict[NonNegativeInt, str]

    _is_segmap_3_dimensions = validator("predictions", allow_reuse=True)(v.is_array_n_dimensions(3))

    @root_validator(pre=True, allow_reuse=True)
    def _check_segmap_shape_and_consistency(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Check that the number of classes equals the depth of the segmentation map.

        Args:
            values (Dict[str, Any]): Dictionary with segmap and classes {param_name: data}.

        Raises:
            ValueError: Raised if there is resolution mismatch between image and mask.

        Returns:
            Dict[str, Any]: Unmodified values parameter passed for further processing.
        """
        if values["predictions"].shape[2] != len(values["index2class"]):
            segmap_depth, nb_classes = values["predictions"].shape, len(values["index2class"])
            raise ValueError(
                f"{cls.__name__}: mismatch between predictions shape {segmap_depth} and classes length {nb_classes}."
            )

        return values

    @property
    def height(self) -> int:
        """Return segmap's height.

        Return:
            int: segmap height.
        """
        return self.predictions.shape[0]

    @property
    def width(self) -> int:
        """Return segmap's width.

        Return:
            int: segmap width.
        """
        return self.predictions.shape[1]

    @property
    def nb_classes(self) -> int:
        """Return the number of classes of the segmentation map (i.e. nb channels).

        Return:
            int: number of classes in the segmentation map.
        """
        return self.predictions.shape[2]

    def __eq__(self, other: object) -> bool:
        """Check if two SegmentationMap objects are equal.

        Args:
            other (object): Second object to compare.

        Returns:
            bool: Comparison result.
        """
        if not isinstance(other, SegmentationMap):
            return False

        return self.index2class == other.index2class and np.allclose(self.predictions, other.predictions)

    def index_of(self, class_name: str) -> int:
        """Get class index based on its name.

        Args:
            class_name (str): Class name

        Raises:
            ValueError: Index of a class

        Returns:
            int: Raised if `class_name` not found in `index2class` dictionary.
        """
        for index, name in self.index2class.items():
            if name == class_name:
                return index

        raise ValueError(f"Index for the `{class_name}` not found")

    def serialize(self) -> Dict[str, Any]:
        """Serialize SegmentationMap object.

        Returns:
            Dict[str, Any]: Serialized object.
        """
        return self.dict(by_alias=True)

    @staticmethod
    def deserialize(data: Dict[str, Any]) -> SegmentationMap:
        """Deserialize SegmentationMap object.

        Args:
            data (Dict[str, Any]): Serialized object to dict.

        Returns:
            SegmentationMap: Deserialized object.
        """
        return SegmentationMap(**data)


class GeometryMask(ImmutableModel):
    """Data holder for the geometry raster."""

    pupil_mask: np.ndarray
    iris_mask: np.ndarray
    eyeball_mask: np.ndarray

    _is_mask_2D = validator("*", allow_reuse=True)(v.is_array_n_dimensions(2))
    _is_binary = validator("*", allow_reuse=True)(v.is_binary)

    @property
    def filled_eyeball_mask(self) -> np.ndarray:
        """Fill eyeball mask.

        Returns:
            np.ndarray: Eyeball mask with filled iris/pupil "holes".
        """
        binary_maps = np.zeros(self.eyeball_mask.shape[:2], dtype=np.uint8)

        binary_maps += self.pupil_mask
        binary_maps += self.iris_mask
        binary_maps += self.eyeball_mask

        return binary_maps.astype(bool)

    @property
    def filled_iris_mask(self) -> np.ndarray:
        """Fill iris mask.

        Returns:
            np.ndarray: Iris mask with filled pupil "holes".
        """
        binary_maps = np.zeros(self.iris_mask.shape[:2], dtype=np.uint8)

        binary_maps += self.pupil_mask
        binary_maps += self.iris_mask

        return binary_maps.astype(bool)

    def serialize(self) -> Dict[str, Any]:
        """Serialize GeometryMask object.

        Returns:
            Dict[str, Any]: Serialized object.
        """
        return self.dict(by_alias=True)

    @staticmethod
    def deserialize(data: Dict[str, Any]) -> GeometryMask:
        """Deserialize GeometryMask object.

        Args:
            data (Dict[str, Any]): Serialized object to dict.

        Returns:
            GeometryMask: Deserialized object.
        """
        return GeometryMask(**data)


class NoiseMask(ImmutableModel):
    """Data holder for the refined geometry masks."""

    mask: np.ndarray

    _is_mask_2D = validator("mask", allow_reuse=True)(v.is_array_n_dimensions(2))
    _is_binary = validator("*", allow_reuse=True)(v.is_binary)

    def serialize(self) -> Dict[str, np.ndarray]:
        """Serialize NoiseMask object.

        Returns:
            Dict[str, np.ndarray]: Serialized object.
        """
        return self.dict(by_alias=True)

    @staticmethod
    def deserialize(data: Dict[str, np.ndarray]) -> NoiseMask:
        """Deserialize NoiseMask object.

        Args:
            data (Dict[str, np.ndarray]): Serialized object to dict.

        Returns:
            NoiseMask: Deserialized object.
        """
        return NoiseMask(**data)


class GeometryPolygons(ImmutableModel):
    """Data holder for the refined geometry polygons. Input np.ndarrays are mandatorily converted to np.float32 dtype for compatibility with some downstream tasks such as MomentsOfArea."""

    pupil_array: np.ndarray
    iris_array: np.ndarray
    eyeball_array: np.ndarray

    _is_list_of_points = validator("*", allow_reuse=True)(v.is_list_of_points)
    _convert_dtype = validator("*", allow_reuse=True)(v.to_dtype_float32)

    @cached_property
    def pupil_diameter(self) -> float:
        """Return pupil diameter.

        Returns:
            float: pupil diameter.
        """
        return estimate_diameter(self.pupil_array)

    @cached_property
    def iris_diameter(self) -> float:
        """Return iris diameter.

        Returns:
            float: iris diameter.
        """
        return estimate_diameter(self.iris_array)

    def serialize(self) -> Dict[str, np.ndarray]:
        """Serialize GeometryPolygons object.

        Returns:
            Dict[str, np.ndarray]: Serialized object.
        """
        return {"pupil": self.pupil_array, "iris": self.iris_array, "eyeball": self.eyeball_array}

    @staticmethod
    def deserialize(data: Dict[str, np.ndarray]) -> GeometryPolygons:
        """Deserialize GeometryPolygons object.

        Args:
            data (Dict[str, np.ndarray]): Serialized object to dict.

        Returns:
            GeometryPolygons: Deserialized object.
        """
        data = {"pupil_array": data["pupil"], "iris_array": data["iris"], "eyeball_array": data["eyeball"]}

        return GeometryPolygons(**data)


class EyeOrientation(ImmutableModel):
    """Data holder for the eye orientation. The angle must be comprised between -pi/2 (included) and pi/2 (excluded)."""

    angle: float = Field(..., ge=-np.pi / 2, lt=np.pi / 2)

    def serialize(self) -> float:
        """Serialize EyeOrientation object.

        Returns:
            float: Serialized object.
        """
        return self.angle

    @staticmethod
    def deserialize(data: float) -> EyeOrientation:
        """Deserialize EyeOrientation object.

        Args:
            data (float): Serialized object to float.

        Returns:
            EyeOrientation: Deserialized object.
        """
        return EyeOrientation(angle=data)


class EyeCenters(ImmutableModel):
    """Data holder for eye's centers."""

    pupil_x: float
    pupil_y: float
    iris_x: float
    iris_y: float

    @property
    def center_distance(self) -> float:
        """Return distance between pupil and iris center.

        Return:
            float: center distance.
        """
        return np.linalg.norm([self.iris_x - self.pupil_x, self.iris_y - self.pupil_y])

    def serialize(self) -> Dict[str, Tuple[float]]:
        """Serialize EyeCenters object.

        Returns:
            Dict[str, Tuple[float]]: Serialized object.
        """
        return {"iris_center": (self.iris_x, self.iris_y), "pupil_center": (self.pupil_x, self.pupil_y)}

    @staticmethod
    def deserialize(data: Dict[str, Tuple[float]]) -> EyeCenters:
        """Deserialize EyeCenters object.

        Args:
            data (Dict[str, Tuple[float]]): Serialized object to dict.

        Returns:
            EyeCenters: Deserialized object.
        """
        data = {
            "pupil_x": data["pupil_center"][0],
            "pupil_y": data["pupil_center"][1],
            "iris_x": data["iris_center"][0],
            "iris_y": data["iris_center"][1],
        }

        return EyeCenters(**data)


class Offgaze(ImmutableModel):
    """Data holder for offgaze score."""

    score: float = Field(..., ge=0.0, le=1.0)

    def serialize(self) -> float:
        """Serialize Offgaze object.

        Returns:
            float: Serialized object.
        """
        return self.score

    @staticmethod
    def deserialize(data: float) -> Offgaze:
        """Deserialize Offgaze object.

        Args:
            data (float): Serialized object to float.

        Returns:
            Offgaze: Deserialized object.
        """
        return Offgaze(score=data)


class Sharpness(ImmutableModel):
    """Data holder for Sharpness score."""

    score: float = Field(..., ge=0.0)

    def serialize(self) -> float:
        """Serialize Sharpness object.

        Returns:
            float: Serialized object.
        """
        return self.score

    @staticmethod
    def deserialize(data: float) -> Sharpness:
        """Deserialize Sharpness object.

        Args:
            data (float): Serialized object to float.

        Returns:
            Sharpness: Deserialized object.
        """
        return Sharpness(score=data)


class PupilToIrisProperty(ImmutableModel):
    """Data holder for pupil-ro-iris ratios."""

    pupil_to_iris_diameter_ratio: float = Field(..., gt=0, lt=1)
    pupil_to_iris_center_dist_ratio: float = Field(..., ge=0, lt=1)

    def serialize(self) -> Dict[str, float]:
        """Serialize PupilToIrisProperty object.

        Returns:
            Dict[str, float]: Serialized object.
        """
        return self.dict(by_alias=True)

    @staticmethod
    def deserialize(data: Dict[str, float]) -> PupilToIrisProperty:
        """Deserialize PupilToIrisProperty object.

        Args:
            data (Dict[str, float]): Serialized object to dict.

        Returns:
            PupilToIrisProperty: Deserialized object.
        """
        return PupilToIrisProperty(**data)


class Landmarks(ImmutableModel):
    """Data holder for eye's landmarks."""

    pupil_landmarks: np.ndarray
    iris_landmarks: np.ndarray
    eyeball_landmarks: np.ndarray

    _is_list_of_points = validator("*", allow_reuse=True)(v.is_list_of_points)

    def serialize(self) -> Dict[str, List[float]]:
        """Serialize Landmarks object.

        Returns:
            Dict[str, List[float]]: Serialized object.
        """
        return {
            "pupil": self.pupil_landmarks.tolist(),
            "iris": self.iris_landmarks.tolist(),
            "eyeball": self.eyeball_landmarks.tolist(),
        }

    @staticmethod
    def deserialize(data: Dict[str, List[float]]) -> Landmarks:
        """Deserialize Landmarks object.

        Args:
            data (Dict[str, List[float]]): Serialized object to dict.

        Returns:
            Landmarks: Deserialized object.
        """
        data = {
            "pupil_landmarks": np.array(data["pupil"]),
            "iris_landmarks": np.array(data["iris"]),
            "eyeball_landmarks": np.array(data["eyeball"]),
        }

        return Landmarks(**data)


class BoundingBox(ImmutableModel):
    """Data holder for eye's bounding box."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    _is_valid_bbox = root_validator(pre=True, allow_reuse=True)(v.is_valid_bbox)

    def serialize(self) -> Dict[str, float]:
        """Serialize BoundingBox object.

        Returns:
            Dict[str, float]: Serialized object.
        """
        return self.dict(by_alias=True)

    @staticmethod
    def deserialize(data: Dict[str, float]) -> BoundingBox:
        """Deserialize BoundingBox object.

        Args:
            data (Dict[str, float]): Serialized object to dict.

        Returns:
            BoundingBox: Deserialized object.
        """
        return BoundingBox(**data)


class NormalizedIris(ImmutableModel):
    """Data holder for the normalized iris images."""

    normalized_image: np.ndarray
    normalized_mask: np.ndarray

    _is_array_2D = validator("*", allow_reuse=True)(v.is_array_n_dimensions(2))
    _is_binary = validator("normalized_mask", allow_reuse=True)(v.is_binary)
    _img_mask_shape_match = root_validator(pre=True, allow_reuse=True)(
        v.are_shapes_equal("normalized_image", "normalized_mask")
    )

    _is_uint8 = validator("normalized_image", allow_reuse=True)(v.is_uint8)

    def serialize(self) -> Dict[str, np.ndarray]:
        """Serialize NormalizedIris object.

        Returns:
            Dict[str, np.ndarray]: Serialized object.
        """
        return self.dict(by_alias=True)

    @staticmethod
    def deserialize(data: Dict[str, np.ndarray]) -> NormalizedIris:
        """Deserialize NormalizedIris object.

        Args:
            data (Dict[str, np.ndarray]): Serialized object to dict.

        Returns:
            NormalizedIris: Deserialized object.
        """
        return NormalizedIris(**data)


class IrisFilterResponse(ImmutableModel):
    """Data holder for filter bank response with associated mask."""

    iris_responses: List[np.ndarray]
    mask_responses: List[np.ndarray]
    iris_code_version: str

    _responses_mask_shape_match = root_validator(pre=True, allow_reuse=True)(
        v.are_all_shapes_equal("iris_responses", "mask_responses")
    )
    _iris_code_version_check = validator("iris_code_version", allow_reuse=True)(v.iris_code_version_check)

    def serialize(self) -> Dict[str, List[np.ndarray]]:
        """Serialize IrisFilterResponse object.

        Returns:
            Dict[str, List[np.ndarray]]: Serialized object.
        """
        return self.dict(by_alias=True)

    @staticmethod
    def deserialize(data: Dict[str, List[np.ndarray]]) -> IrisFilterResponse:
        """Deserialize IrisFilterResponse object.

        Args:
            data (Dict[str, List[np.ndarray]]): Serialized object to dict.

        Returns:
            IrisFilterResponse: Deserialized object.
        """
        return IrisFilterResponse(**data)


class IrisTemplate(ImmutableModel):
    """Data holder for final iris template with mask."""

    iris_codes: List[np.ndarray]
    mask_codes: List[np.ndarray]
    iris_code_version: str

    _responses_mask_shape_match = root_validator(pre=True, allow_reuse=True)(
        v.are_all_shapes_equal("iris_codes", "mask_codes")
    )
    _is_binary = validator("iris_codes", "mask_codes", allow_reuse=True, each_item=True)(v.is_binary)
    _iris_code_version_check = validator("iris_code_version", allow_reuse=True)(v.iris_code_version_check)

    def serialize(self) -> Dict[str, bytes]:
        """Serialize IrisTemplate object.

        Returns:
            Dict[str, bytes]: Serialized object.
        """
        old_format_iris_codes, old_format_mask_codes = self.convert2old_format()

        return {
            "iris_codes": base64_encode_array(old_format_iris_codes).decode("utf-8"),
            "mask_codes": base64_encode_array(old_format_mask_codes).decode("utf-8"),
            "iris_code_version": self.iris_code_version,
        }

    def convert2old_format(self) -> List[np.ndarray]:
        """Convert an old tempalte format and the associated iris code version into an IrisTemplate object.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Old engines pipeline object Tuple with (iris_codes, mask_codes).
        """
        return (IrisTemplate.new_to_old_format(self.iris_codes), IrisTemplate.new_to_old_format(self.mask_codes))

    @staticmethod
    def deserialize(
        serialized_template: dict[str, Union[np.ndarray, str]], array_shape: Tuple = (16, 256, 2, 2)
    ) -> IrisTemplate:
        """Deserialize a dict with iris_codes, mask_codes and iris_code_version into an IrisTemplate object.

        Args:
            serialized_template (dict[str, Union[np.ndarray, str]]): Serialized object to dict.
            array_shape (Tuple, optional): Shape of the iris code. Defaults to (16, 256, 2, 2).

        Returns:
            IrisTemplate: Serialized object.
        """
        return IrisTemplate.convert_to_new_format(
            iris_codes=base64_decode_array(serialized_template["iris_codes"], array_shape=array_shape),
            mask_codes=base64_decode_array(serialized_template["mask_codes"], array_shape=array_shape),
            iris_code_version=serialized_template["iris_code_version"],
        )

    @staticmethod
    def convert_to_new_format(iris_codes: np.ndarray, mask_codes: np.ndarray, iris_code_version: str) -> IrisTemplate:
        """Convert an old template format and the associated iris code version into an IrisTemplate object.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Old engines pipeline object Tuple with (iris_codes, mask_codes).
        """
        return IrisTemplate(
            iris_codes=IrisTemplate.old_to_new_format(iris_codes),
            mask_codes=IrisTemplate.old_to_new_format(mask_codes),
            iris_code_version=iris_code_version,
        )

    @staticmethod
    def new_to_old_format(array: List[np.ndarray]) -> np.ndarray:
        """Convert new iris template format to old iris template format.
            - New format is a list of arrays, each of shape (height_i, width_i, 2). The length of the list is nb_wavelets.
                This enable having different convolution layout for each wavelet.
            - Old format is a numpy array of shape (height, width, nb_wavelets, 2)

        Args:
            codes (List[np.ndarray]): New format iris/mask codes.

        Returns:
            np.ndarray: Old format codes.

        Raises:
            ValueError: Raised if not all codes have the same shape. In this case, the IrisTemplate cannot be converted to the old format.
        """
        if not all([code.shape == array[0].shape for code in array]):
            raise ValueError("All codes must have the same shape to be converted to the old format.")
        return np.stack(array).transpose(1, 2, 0, 3)

    @staticmethod
    def old_to_new_format(array: np.ndarray) -> List[np.ndarray]:
        """Convert old iris template format to new iris template format.
            - Old format is a list of arrays, each of shape (height_i, width_i, 2). The length of the list is nb_wavelets.
                This enable having different convolution layout for each wavelet.
            - New format is a numpy array of shape (height, width, nb_wavelets, 2)

        Args:
            codes (List[np.ndarray]): Old format iris/mask codes.

        Returns:
            np.ndarray: New format codes.
        """
        return [array[:, :, i, :] for i in range(array.shape[2])]


class WeightedIrisTemplate(IrisTemplate):
    """
    Extends IrisTemplate to include per-bit reliability weights.
    """

    weights: List[np.ndarray] = Field(..., description="List of weight matrices per wavelet, shape matches iris_codes.")

    def as_iris_template(self) -> IrisTemplate:
        """Convert a WeightedIrisTemplate to an IrisTemplate.

        Returns:
            IrisTemplate: IrisTemplate object.
        """
        return IrisTemplate(
            iris_codes=self.iris_codes,
            mask_codes=self.mask_codes,
            iris_code_version=self.iris_code_version,
        )

    @staticmethod
    def from_iris_template(iris_template: IrisTemplate, weights: List[np.ndarray]) -> WeightedIrisTemplate:
        """Create a WeightedIrisTemplate from an IrisTemplate and a list of weights.

        Args:
            iris_template (IrisTemplate): IrisTemplate to convert.
            weights (List[np.ndarray]): List of weight matrices per wavelet, shape matches iris_codes.

        Returns:
            WeightedIrisTemplate: WeightedIrisTemplate object.
        """
        return WeightedIrisTemplate(
            iris_codes=iris_template.iris_codes,
            mask_codes=iris_template.mask_codes,
            weights=weights,
            iris_code_version=iris_template.iris_code_version,
        )

    @root_validator(pre=True)
    def check_weights_shape_and_length(cls, values):
        iris_codes = values.get("iris_codes")
        weights = values.get("weights")
        if iris_codes is None or weights is None:
            raise ValueError("iris_codes and weights must both be provided")
        if len(weights) != len(iris_codes):
            raise ValueError(f"weights and iris_codes must have same length. Got {len(weights)} and {len(iris_codes)}.")
        for idx, (w, c) in enumerate(zip(weights, iris_codes)):
            if w.shape != c.shape:
                raise ValueError(f"Shape mismatch at wavelet {idx}: weight shape {w.shape}, code shape {c.shape}")
            if not np.all(w >= 0):
                raise ValueError(f"All weights must be >= 0 (found negative at wavelet {idx})")
        return values

    @validator("weights", each_item=True)
    def weights_are_valid(cls, v):
        if not np.issubdtype(v.dtype, np.floating):
            raise TypeError(f"Weight must be float array, got {v.dtype}")
        return v

    def serialize(self) -> Dict[str, Any]:
        """Serialize WeightedIrisTemplate object.

        Returns:
            Dict[str, bytes]: Serialized object.
        """
        old_format_iris_codes, old_format_mask_codes, old_format_weights = self.convert2old_format()

        return {
            "iris_codes": base64_encode_array(old_format_iris_codes).decode("utf-8"),
            "mask_codes": base64_encode_array(old_format_mask_codes).decode("utf-8"),
            "weights": base64_encode_float_array(old_format_weights).decode("utf-8"),
            "iris_code_version": self.iris_code_version,
        }

    @staticmethod
    def deserialize(
        serialized_template: dict[str, Union[np.ndarray, str]], array_shape: Tuple = (16, 256, 2, 2)
    ) -> WeightedIrisTemplate:
        """Deserialize a dict with iris_codes, mask_codes and iris_code_version into an IrisTemplate object.

        Args:
            serialized_template (dict[str, Union[np.ndarray, str]]): Serialized object to dict.
            array_shape (Tuple, optional): Shape of the iris code. Defaults to (16, 256, 2, 2).

        Returns:
            WeightedIrisTemplate: Serialized object.
        """
        return WeightedIrisTemplate.convert_to_new_format(
            iris_codes=base64_decode_array(serialized_template["iris_codes"], array_shape=array_shape),
            mask_codes=base64_decode_array(serialized_template["mask_codes"], array_shape=array_shape),
            weights=base64_decode_float_array(serialized_template["weights"], array_shape=array_shape),
            iris_code_version=serialized_template["iris_code_version"],
        )

    def convert2old_format(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert an old tempalte format and the associated iris code version into an IrisTemplate object.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Old engines pipeline object Tuple with (iris_codes, mask_codes, weights).
        """
        return (
            IrisTemplate.new_to_old_format(self.iris_codes),
            IrisTemplate.new_to_old_format(self.mask_codes),
            IrisTemplate.new_to_old_format(self.weights),
        )

    @staticmethod
    def convert_to_new_format(
        iris_codes: np.ndarray, mask_codes: np.ndarray, weights: np.ndarray, iris_code_version: str
    ) -> WeightedIrisTemplate:
        """Convert an old template format and the associated iris code version into an IrisTemplate object.

        Returns:
            WeightedIrisTemplate: Serialized object in new format.
        """
        return WeightedIrisTemplate(
            iris_codes=IrisTemplate.old_to_new_format(iris_codes),
            mask_codes=IrisTemplate.old_to_new_format(mask_codes),
            weights=IrisTemplate.old_to_new_format(weights),
            iris_code_version=iris_code_version,
        )


class EyeOcclusion(ImmutableModel):
    """Data holder for the eye occlusion."""

    visible_fraction: float = Field(..., ge=-0.0, le=1.0)

    def serialize(self) -> float:
        """Serialize EyeOcclusion object.

        Returns:
            float: Serialized object.
        """
        return self.visible_fraction

    @staticmethod
    def deserialize(data: float) -> EyeOcclusion:
        """Deserialize EyeOcclusion object.

        Args:
            data (float): Serialized object to float.

        Returns:
            EyeOcclusion: Deserialized object.
        """
        return EyeOcclusion(visible_fraction=data)


class OutputFieldSpec(BaseModel):
    """
    Specification for a single output field in the pipeline result.

    Attributes:
        key (str): The name of the field in the output dictionary.
        extractor (Callable[[PipelineCallTraceStorage], Any]): A function that takes a PipelineCallTraceStorage and returns the raw value.
        safe_serialize (bool): If True, apply __safe_serialize to the extracted value before returning it.
    """

    key: str
    extractor: Callable[[PipelineCallTraceStorage], Any]
    safe_serialize: bool = False


class DistanceMatrix(ImmutableModel):
    """Data holder for a distance matrix."""

    data: Dict[Tuple[int, int], float]

    def get(self, i: int, j: int) -> float:
        """Get the distance between two templates.

        Args:
            i (int): Index of the first template.
            j (int): Index of the second template.

        Returns:
            float: Distance between the two templates.
        """
        key = (min(i, j), max(i, j))
        return self.data[key]

    def to_numpy(self) -> np.ndarray:
        """Convert the distance matrix to a numpy array.

        Returns:
            np.ndarray: Distance matrix.
        """
        n = self.nb_templates
        mat = np.zeros((n, n), dtype=float)
        for (i, j), value in self.data.items():
            mat[i, j] = value
            mat[j, i] = value  # symmetry
        return mat

    def to_matrix(self) -> np.ndarray:
        """Convert the distances to a symmetric matrix.

        Returns:
            np.ndarray: Distance matrix.
        """
        return self.to_numpy()

    def __len__(self) -> int:
        """Return the number of distances in the matrix.

        Returns:
            int: Number of distances.
        """
        return len(self.data)

    @property
    def nb_templates(self) -> int:
        """Number of unique template indices present in the matrix.

        Returns:
            int: Number of unique template indices.
        """
        indices = set()
        for i, j in self.data.keys():
            indices.add(i)
            indices.add(j)
        return len(indices)

    def serialize(self) -> Dict[Tuple[int, int], float]:
        """Serialize DistanceMatrix object.

        Returns:
            Dict[Tuple[int, int], float]: Serialized object.
        """
        return self.data

    @staticmethod
    def deserialize(data: Dict[Tuple[int, int], float]) -> DistanceMatrix:
        """Deserialize DistanceMatrix object.

        Returns:
            DistanceMatrix: Deserialized object.
        """
        return DistanceMatrix(data=data)


class AlignedTemplates(ImmutableModel):
    """Data holder for aligned templates and Hamming distances between them.

    Args:
        aligned_templates (List[IrisTemplate]): List of aligned Iris templates.
        distances (Dict[Tuple[int, int], float]): Dictionary of Hamming distances between Iris templates.
        reference_template_id (int): Index of the reference template.
    """

    templates: List[IrisTemplate]
    distances: DistanceMatrix
    reference_template_id: int

    @root_validator(pre=True, allow_reuse=True)
    def _check_distances_match_templates(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Check that the number of distances corresponds to the number of aligned templates.

        Args:
            values (Dict[str, Any]): Dictionary containing the field values.

        Returns:
            Dict[str, Any]: The validated values.

        Raises:
            ValueError: If the number of distances doesn't match the number of templates.
        """
        templates = values.get("templates")
        distances = values.get("distances")

        if templates is not None and distances is not None:
            nb_templates = len(templates)
            nb_distances = distances.nb_templates

            if (nb_distances != nb_templates) & (nb_templates > 1):
                raise ValueError(
                    f"Number of templates ({nb_templates}) does not match number of distances ({nb_distances}). "
                    f"Expected {nb_templates} templates but found {nb_distances} in distance matrix."
                )

        return values

    @root_validator(pre=True, allow_reuse=True)
    def _check_reference_template_id_valid(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Check that the reference_template_id is within the valid range of aligned templates indices.

        Args:
            values (Dict[str, Any]): Dictionary containing the field values.

        Returns:
            Dict[str, Any]: The validated values.

        Raises:
            ValueError: If the reference_template_id is out of range.
        """
        templates = values.get("templates")
        reference_template_id = values.get("reference_template_id")

        if templates is not None and reference_template_id is not None:
            nb_templates = len(templates)

            if reference_template_id < 0:
                raise ValueError(
                    f"reference_template_id ({reference_template_id}) cannot be negative. "
                    f"Must be between 0 and {nb_templates - 1}."
                )

            if reference_template_id >= nb_templates:
                raise ValueError(
                    f"reference_template_id ({reference_template_id}) is out of range. "
                    f"Must be between 0 and {nb_templates - 1} for {nb_templates} templates."
                )

        return values

    @property
    def reference_template(self) -> IrisTemplate:
        """Get the reference template.

        Returns:
            IrisTemplate: Reference template.
        """
        return self.templates[self.reference_template_id]

    def get_distance(self, i: int, j: int) -> float:
        """Get the distance between two templates.

        Args:
            i (int): Index of the first template.
            j (int): Index of the second template.

        Returns:
            float: Distance between the two templates.
        """
        return self.distances.get(i, j)

    def __len__(self) -> int:
        """Return the number of aligned templates.

        Returns:
            int: Number of aligned templates.
        """
        return len(self.templates)

    def serialize(self) -> Dict[str, Any]:
        """Serialize AlignedTemplates object.

        Returns:
            Dict[str, Any]: Serialized object.
        """
        return {
            "templates": [template.serialize() for template in self.templates],
            "distances": self.distances.serialize(),
            "reference_template_id": self.reference_template_id,
        }

    @staticmethod
    def deserialize(data: Dict[str, Any], array_shape: Tuple[int, int, int, int] = (16, 256, 2, 2)) -> AlignedTemplates:
        """Deserialize AlignedTemplates object.

        Args:
            data (Dict[str, Any]): Serialized object to dict.
            array_shape (Tuple[int, int, int, int], optional): Shape of the iris code. Defaults to (16, 256, 2, 2).

        Returns:
            AlignedTemplates: Deserialized object.
        """
        return AlignedTemplates(
            templates=[IrisTemplate.deserialize(template, array_shape) for template in data["templates"]],
            distances=DistanceMatrix.deserialize(data["distances"]),
            reference_template_id=data["reference_template_id"],
        )
