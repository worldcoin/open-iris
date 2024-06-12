*Matching entities* tutorial
================================

This subpage will walk you through the basics of how to use matchers available in the ``iris`` package. From it you will learn how to:
- Use the ``HammingDistanceMatcher`` matcher to compute distance between two eyes.

1. Use the ``HammingDistanceMatcher`` matcher to compute distance between two eyes.
------------------------------------------------------------------------------------------------

Load all IR images with ``opencv-python`` package.

.. code-block:: python

    import cv2

    subject1_first_image = cv2.imread("./subject1_first_image.png", cv2.IMREAD_GRAYSCALE)
    subject1_second_image = cv2.imread("./subject1_second_image.png", cv2.IMREAD_GRAYSCALE)
    subject2_image = cv2.imread("./subject2_image.png", cv2.IMREAD_GRAYSCALE)

Create ``IRISPipeline`` object and compute ``IrisTemplates`` for all images.

.. code-block:: python

    import iris

    iris_pipeline = iris.IRISPipeline()

    output_1 = iris_pipeline(subject1_first_image, eye_side="left")
    subject1_first_code = output_1["iris_template"]

    output_2 = iris_pipeline(subject1_second_image, eye_side="left")
    subject1_second_code = output_2["iris_template"]

    output_3 = iris_pipeline(subject2_image, eye_side="left")
    subject2_code = output_3["iris_template"]

Create a ``HammingDistanceMatcher`` matcher object.

.. code-block:: python

    def __init__(
        self,
        rotation_shift: int = 15,
        nm_dist: Optional[confloat(ge=0, le=1, strict=True)] = None,
        weights: Optional[List[np.ndarray]] = None,
    ) -> None:
        """Assign parameters.

        Args:
            rotation_shift (int): rotations allowed in matching, converted to shifts in columns. Defaults to 15.
            nm_dist (Optional[confloat(ge=0, le = 1, strict=True)]): nonmatch distance used for normalized HD. Optional paremeter for normalized HD. Defaults to None.
            weights (Optional[List[np.ndarray]]): list of weights table. Optional paremeter for weighted HD. Defaults to None.
        """

.. code-block:: python

    matcher = iris.HammingDistanceMatcher()

Call ``run`` method and provide two ``IrisTemplates`` to compute distances.

.. code-block:: python

    def run(self, template_probe: IrisTemplate, template_gallery: IrisTemplate) -> float:

.. code-block:: python

    same_subjects_distance = matcher.run(subject1_first_code, subject1_second_code)
    different_subjects_distance = matcher.run(subject1_first_code, subject2_code)

**Thank you for making it to the end of this tutorial!**
