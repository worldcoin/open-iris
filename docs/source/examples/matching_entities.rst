*Matching entities* tutorial
================================

This subpage will walk you through the basics of how to use matchers available in the ``iris`` package. From it you will learn how to:
- Use the ``HashBasedMatcher`` matcher to compute unique identifiers between two eyes.

1. Use the ``HashBasedMatcher`` matcher to compute unique identifiers between two eyes.
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

Create a ``HashBasedMatcher`` matcher object.

.. code-block:: python

    def __init__(
        self,
        rotation_shift: int = 0,
        hash_bits: int = 40,
    ) -> None:
        """Assign parameters.

       Args:
            rotation_shift (int): Kept for interface compatibility, not used in hash-based approach. Defaults to 0.
            hash_bits (int): Number of bits to extract from hash. Defaults to 40.
        """

.. code-block:: python

    matcher = iris.HashBasedMatcher()

Call ``run`` method and provide two ``IrisTemplates`` to compute unique identifiers.

.. code-block:: python

    def run(self, template_probe: IrisTemplate, template_gallery: IrisTemplate) -> float:

.. code-block:: python

    same_subjects_match = matcher.run(subject1_first_code, subject1_second_code)
    different_subjects_match = matcher.run(subject1_first_code, subject2_code)

    # Get unique identifiers
    subject1_first_id = matcher.get_unique_id(subject1_first_code)
    subject1_second_id = matcher.get_unique_id(subject1_second_code)
    subject2_id = matcher.get_unique_id(subject2_code)

    print(f"Subject 1 first ID: {subject1_first_id}")
    print(f"Subject 1 second ID: {subject1_second_id}")
    print(f"Subject 2 ID: {subject2_id}")
    print(f"Same subjects match: {same_subjects_match == 0.0}")
    print(f"Different subjects match: {different_subjects_match == 0.0}")

**Thank you for making it to the end of this tutorial!**
