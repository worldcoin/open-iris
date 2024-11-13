from __future__ import annotations

from typing import Any, Dict, List, Literal, Tuple, Union

import numpy as np
from pydantic import Field, NonNegativeInt, root_validator, validator

from iris.io import validators as v
from iris.io.class_configs import ImmutableModel
from iris.utils.base64_encoding import base64_decode_array, base64_encode_array
from iris.utils.math import estimate_diameter


class IRImage(ImmutableModel):
    """Data holder for input IR image."""

    img_data: np.ndarray
    eye_side: Literal["left", "right"]
    bboxes: np.ndarray  # used for SAM

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

    @property
    def pupil_diameter(self) -> float:
        """Return pupil diameter.

        Returns:
            float: pupil diameter.
        """
        return estimate_diameter(self.pupil_array)

    @property
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
