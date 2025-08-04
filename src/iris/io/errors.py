class EncoderError(Exception):
    """Encoder module Error class."""

    pass


class EyeCentersEstimationError(Exception):
    """EyeOrientationEstimation module Error class."""

    pass


class EyeOrientationEstimationError(Exception):
    """EyeOrientationEstimation module Error class."""

    pass


class OffgazeEstimationError(Exception):
    """OffgazeEstimation module Error class."""

    pass


class BoundingBoxEstimationError(Exception):
    """BoundingBoxEstimationError module Error class."""

    pass


class LandmarkEstimationError(Exception):
    """LandmarkEstimationError module Error class."""

    pass


class OcclusionError(Exception):
    """EyeOrientationEstimation module Error class."""

    pass


class PupilIrisPropertyEstimationError(Exception):
    """PupilIrisPropertyEstimation module Error class."""

    pass


class Pupil2IrisValidatorErrorDilation(Exception):
    """Pupil2IrisValidatorErrorDilation error class."""

    pass


class Pupil2IrisValidatorErrorConstriction(Exception):
    """Pupil2IrisValidatorErrorConstriction error class."""

    pass


class Pupil2IrisValidatorErrorOffcenter(Exception):
    """Pupil2IrisValidatorErrorOffcenter error class."""

    pass


class GeometryEstimationError(Exception):
    """GeometryEstimation module Error class."""

    pass


class GeometryRefinementError(Exception):
    """GeometryRefinementError error class."""

    pass


class ImageFilterError(Exception):
    """ImageFilter's base and subclasses error class."""

    pass


class ProbeSchemaError(Exception):
    """ProbeSchema's base and subclasses error class."""

    pass


class NormalizationError(Exception):
    """Normalization module Error class."""

    pass


class EyeCentersInsideImageValidatorError(Exception):
    """EyeCentersInsideImageValidatorError error class."""

    pass


class ExtrapolatedPolygonsInsideImageValidatorError(Exception):
    """ExtrapolatedPolygonsInsideImageValidatorError error class."""

    pass


class IsPupilInsideIrisValidatorError(Exception):
    """IsPupilInsideIrisValidator error class."""

    pass


class VectorizationError(Exception):
    """Vectorization module Error class."""

    pass


class SharpnessEstimationError(Exception):
    """SharpnessEstimation Error class."""

    pass


class MaskTooSmallError(Exception):
    """Mask is too small Error class."""

    pass


class MatcherError(Exception):
    """Matcher module Error class."""

    pass


class IRISPipelineError(Exception):
    """IRIS Pipeline module Error class."""

    pass


class TemplateAggregationCompatibilityError(Exception):
    """Template aggregation compatibility validation Error class."""

    pass


class TemplatesAggregationPipelineError(Exception):
    """TemplatesAggregationPipeline module Error class."""

    pass


class IdentityValidationError(Exception):
    """Identity validation Error class."""

    pass
