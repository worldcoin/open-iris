Running inference
================================

A simple inference run can be achieved by running source code below.

.. code-block:: python

    import cv2
    import iris

    # 1. Create IRISPipeline object
    iris_pipeline = iris.IRISPipeline()

    # 2. Load IR image of an eye
    img_pixels = cv2.imread("/path/to/ir/image", cv2.IMREAD_GRAYSCALE)

    # 3. Perform inference
    # Options for the `eye_side` argument are: ["left", "right"]
    output = iris_pipeline(img_data=img_pixels, eye_side="left")

To fully explore and understand the extensive capabilities of the iris package, visit the `Examples` subpages. Here, you'll find a collection of Jupyter Notebooks that serve as valuable resources, offering practical guides and real-world examples to provide a comprehensive insight into the rich functionalities and potential applications of the ``iris`` package.

