*Configuring custom pipeline* tutorial
========================================

This subpage will walk you through you the steps you have to take to configure your custom ``IRISPipeline``. From it you will learn how to:

* Configure ``IRISPipeline`` algorithms parameters.
* Configure ``IRISPipeline`` graph.
* Implement your own node with ``Algorithm`` class and introduce them into ``IRISPipeline`` graph.

1. Configure ``IRISPipeline`` algorithms parameters.
------------------------------------------------------

To create the ``IRISPipeline`` object with modified ``Algorithm`` parameters, we have to understand a few things about ``IRISPipeline`` orchestration system.

When the ``IRISPipeline`` pipeline is created with default parameters, it's graph is parsed from a default YAML file that is available in ``/src/iris/pipelines/confs/pipeline.yaml`` YAML file. The content of that file presents as follow.

.. code-block:: yaml

    metadata:
    pipeline_name: iris_pipeline
    iris_version: 1.1.1

The top YAML file contains ``IRISPipeline`` metadata, used to both describe ``IRISPipeline`` and specify package parameters that are later used to verify compatibility between ``iris`` package version/release and later, specified in the ``pipeline`` YAML file section, pipeline's graph.

.. code-block:: yaml

    pipeline:
        - name: segmentation
            algorithm:
            class_name: iris.MultilabelSegmentation.create_from_hugging_face
            params: {}
            inputs:
            - name: image
                source_node: input
            callbacks:
        - name: segmentation_binarization
            algorithm:
                class_name: iris.MultilabelSegmentationBinarization
                params: {}
            inputs:
            - name: segmentation_map
                source_node: segmentation
            callbacks:
        ...

The ``pipeline`` subsection contains a list of ``IRISPipeline`` nodes. The node definition has to contain following keys:

* ``name`` - that's node metadata information about node name. It's used later to define connections with other defined nodes. Also, it's worth to notice that the ``name`` key is later used by ``PipelineCallTraceStorage`` to store and return different intermediate results.
* ``algorithm`` - that's a key that contains a definition of a Python object that implements an algorithm we want to use in our pipeline.
* ``algorithms.class_name`` - a Python object class name that implements ``iris.Algorithm`` interface (more information about ``Algorithm`` class will be provided in section 3 of this tutorial). Please note, that defined here Python object must be importable by Python interpreter. That means that ``Algorithm`` implementation doesn't have to be implemented within ``iris`` package. User may implement or import it from any external library. The only constraint is that ``Algorithm`` interface must be satisfied to make everything compatible.
* ``algorithms.params`` - that key defined a dictionary that contains all ``__init__`` parameters of a given node - ``Algorithm`` object. List of parameters of nodes available in the ``iris`` package with their descriptions can be found in project documentation.
* ``inputs`` - that key defined a list of inputs to node's ``run`` method - connections between node within pipeline graph. A single input record has to contain following keys: ``["name", "source_node"]``. Optionally, an ``inputs`` record can contain an ``index`` key. It's used whenever input node returns a tuple/list of objects and user wants to extract a certain output to be provided to ``run`` method of currently defined node. An example of a node definition that utilized ``index`` can look like follow:

.. code-block:: yaml

    - name: vectorization
        algorithm:
          class_name: iris.ContouringAlgorithm
          params: {}
        inputs:
          - name: geometry_mask
            source_node: segmentation_binarization
            index: 0
        callbacks:

* ``inputs.name`` - the ``Algorithm`` ``run`` method argument name that is meant to be filled with the output from the ``source_name``.
* ``inputs.source_name`` - a name of node that outputs input to currently defined node.
* ``callbacks`` - a key that defines a list of possible ``iris.Callback`` object of a node. That key requires from an ``Algorithm`` object to allow callback plug in. User can allow that behaviour when specifying ``callbacks`` argument of the ``__init__`` method of particular ``Algorithm``.

*NOTE*: Nodes has to be defined consecutively with the order they appear within pipeline. That means that specifying ``source_name`` to the node which definition appears later within YAML file will cause exception being raised when instantiating pipeline.

A default pipeline configuration specified within YAML file can be found in `/src/iris/pipelines/confs/pipeline.yaml <https://github.com/worldcoin/open-iris/blob/main/src/iris/pipelines/confs/pipeline.yaml>`_.

Other then YAML file, user may defined and provide to ``__init__`` method a Python dictionary with similar structure as described above YAML file.

Below examples shows how to modify ``iris.MultilabelSegmentationBinarization`` algorithm thresholds to use other than specified by default ``0.5``. The ``iris.MultilabelSegmentationBinarization`` ``__init__`` method is defined as follow:

.. code-block:: python

    class MultilabelSegmentationBinarization(Algorithm):
        def __init__(
            self,
            eyeball_threshold: float = 0.5,
            iris_threshold: float = 0.5,
            pupil_threshold: float = 0.5,
            eyelashes_threshold: float = 0.5,
        ) -> None:
            ...
        ...

First let's intantiate ``IRISPipeline`` with default configuration and see ``iris.MultilabelSegmentationBinarization`` threshold values.

.. code-block:: python

    default_pipeline_conf = {
        "metadata": {"pipeline_name": "iris_pipeline", "iris_version": "1.1.1"},
        "pipeline": [
            {
                "name": "segmentation",
                "algorithm": {"class_name": "iris.MultilabelSegmentation.create_from_hugging_face", "params": {}},
                "inputs": [{"name": "image", "source_node": "input"}],
                "callbacks": None,
            },
    ############################### A NODE, WHICH PARAMETERS WE WANT TO MODIFY ################################
            {
                "name": "segmentation_binarization",
                "algorithm": {"class_name": "iris.MultilabelSegmentationBinarization", "params": {}},
                "inputs": [{"name": "segmentation_map", "source_node": "segmentation"}],
                "callbacks": None,
            },
    ############################################################################################################
            {
                "name": "vectorization",
                "algorithm": {"class_name": "iris.ContouringAlgorithm", "params": {}},
                "inputs": [{"name": "geometry_mask", "source_node": "segmentation_binarization", "index": 0}],
                "callbacks": None,
            },
            {
                "name": "specular_reflection_detection",
                "algorithm": {"class_name": "iris.SpecularReflectionDetection", "params": {}},
                "inputs": [{"name": "ir_image", "source_node": "input"}],
                "callbacks": None,
            },
            {
                "name": "interpolation",
                "algorithm": {"class_name": "iris.ContourInterpolation", "params": {}},
                "inputs": [{"name": "polygons", "source_node": "vectorization"}],
                "callbacks": None,
            },
            {
                "name": "distance_filter",
                "algorithm": {"class_name": "iris.ContourPointNoiseEyeballDistanceFilter", "params": {}},
                "inputs": [
                    {"name": "polygons", "source_node": "interpolation"},
                    {"name": "geometry_mask", "source_node": "segmentation_binarization", "index": 1},
                ],
                "callbacks": None,
            },
            {
                "name": "eye_orientation",
                "algorithm": {"class_name": "iris.MomentOfArea", "params": {}},
                "inputs": [{"name": "geometries", "source_node": "distance_filter"}],
                "callbacks": None,
            },
            {
                "name": "eye_center_estimation",
                "algorithm": {"class_name": "iris.BisectorsMethod", "params": {}},
                "inputs": [{"name": "geometries", "source_node": "distance_filter"}],
                "callbacks": None,
            },
            {
                "name": "smoothing",
                "algorithm": {"class_name": "iris.Smoothing", "params": {}},
                "inputs": [
                    {"name": "polygons", "source_node": "distance_filter"},
                    {"name": "eye_centers", "source_node": "eye_center_estimation"},
                ],
                "callbacks": None,
            },
            {
                "name": "geometry_estimation",
                "algorithm": {
                    "class_name": "iris.FusionExtrapolation",
                    "params": {
                        "circle_extrapolation": {"class_name": "iris.LinearExtrapolation", "params": {"dphi": 0.703125}},
                        "ellipse_fit": {"class_name": "iris.LSQEllipseFitWithRefinement", "params": {"dphi": 0.703125}},
                        "algorithm_switch_std_threshold": 3.5,
                    },
                },
                "inputs": [
                    {"name": "input_polygons", "source_node": "smoothing"},
                    {"name": "eye_center", "source_node": "eye_center_estimation"},
                ],
                "callbacks": None,
            },
            {
                "name": "pupil_to_iris_property_estimation",
                "algorithm": {"class_name": "iris.PupilIrisPropertyCalculator", "params": {}},
                "inputs": [
                    {"name": "geometries", "source_node": "geometry_estimation"},
                    {"name": "eye_centers", "source_node": "eye_center_estimation"},
                ],
                "callbacks": None,
            },
            {
                "name": "offgaze_estimation",
                "algorithm": {"class_name": "iris.EccentricityOffgazeEstimation", "params": {}},
                "inputs": [{"name": "geometries", "source_node": "geometry_estimation"}],
                "callbacks": None,
            },
            {
                "name": "occlusion90_calculator",
                "algorithm": {"class_name": "iris.OcclusionCalculator", "params": {"quantile_angle": 90.0}},
                "inputs": [
                    {"name": "noise_mask", "source_node": "segmentation_binarization", "index": 1},
                    {"name": "extrapolated_polygons", "source_node": "geometry_estimation"},
                    {"name": "eye_orientation", "source_node": "eye_orientation"},
                    {"name": "eye_centers", "source_node": "eye_center_estimation"},
                ],
                "callbacks": None,
            },
            {
                "name": "occlusion30_calculator",
                "algorithm": {"class_name": "iris.OcclusionCalculator", "params": {"quantile_angle": 30.0}},
                "inputs": [
                    {"name": "noise_mask", "source_node": "segmentation_binarization", "index": 1},
                    {"name": "extrapolated_polygons", "source_node": "geometry_estimation"},
                    {"name": "eye_orientation", "source_node": "eye_orientation"},
                    {"name": "eye_centers", "source_node": "eye_center_estimation"},
                ],
                "callbacks": None,
            },
            {
                "name": "noise_masks_aggregation",
                "algorithm": {"class_name": "iris.NoiseMaskUnion", "params": {}},
                "inputs": [
                    {
                        "name": "elements",
                        "source_node": [
                            {"name": "segmentation_binarization", "index": 1},
                            {"name": "specular_reflection_detection"},
                        ],
                    }
                ],
                "callbacks": None,
            },
            {
                "name": "normalization",
                "algorithm": {"class_name": "iris.PerspectiveNormalization", "params": {}},
                "inputs": [
                    {"name": "image", "source_node": "input"},
                    {"name": "noise_mask", "source_node": "noise_masks_aggregation"},
                    {"name": "extrapolated_contours", "source_node": "geometry_estimation"},
                    {"name": "eye_orientation", "source_node": "eye_orientation"},
                ],
                "callbacks": None,
            },
            {
                "name": "filter_bank",
                "algorithm": {
                    "class_name": "iris.ConvFilterBank",
                    "params": {
                        "filters": [
                            {
                                "class_name": "iris.GaborFilter",
                                "params": {
                                    "kernel_size": [41, 21],
                                    "sigma_phi": 7,
                                    "sigma_rho": 6.13,
                                    "theta_degrees": 90.0,
                                    "lambda_phi": 28.0,
                                    "dc_correction": True,
                                    "to_fixpoints": True,
                                },
                            },
                            {
                                "class_name": "iris.GaborFilter",
                                "params": {
                                    "kernel_size": [17, 21],
                                    "sigma_phi": 2,
                                    "sigma_rho": 5.86,
                                    "theta_degrees": 90.0,
                                    "lambda_phi": 8,
                                    "dc_correction": True,
                                    "to_fixpoints": True,
                                },
                            },
                        ],
                        "probe_schemas": [
                            {"class_name": "iris.RegularProbeSchema", "params": {"n_rows": 16, "n_cols": 256}},
                            {"class_name": "iris.RegularProbeSchema", "params": {"n_rows": 16, "n_cols": 256}},
                        ],
                    },
                },
                "inputs": [{"name": "normalization_output", "source_node": "normalization"}],
                "callbacks": None,
            },
            {
                "name": "encoder",
                "algorithm": {"class_name": "iris.IrisEncoder", "params": {}},
                "inputs": [{"name": "response", "source_node": "filter_bank"}],
                "callbacks": None,
            },
            {
                "name": "bounding_box_estimation",
                "algorithm": {"class_name": "iris.IrisBBoxCalculator", "params": {}},
                "inputs": [
                    {"name": "ir_image", "source_node": "input"},
                    {"name": "geometry_polygons", "source_node": "geometry_estimation"},
                ],
                "callbacks": None,
            },
        ],
    }

Instantiate ``IRISPipeline`` object.

.. code-block:: python

    iris_pipeline = iris.IRISPipeline(config=default_pipeline_conf)

Print ``iris.MultilabelSegmentationBinarization`` threshold values.

.. code-block:: python

    def print_segmentation_binarization_thresholds():
        binarization_node = [node for node_name, node in iris_pipeline.nodes.items() if node_name == "segmentation_binarization"]

        assert len(binarization_node) == 1

        binarization_node = binarization_node[0]
        print(binarization_node.params)

    print_segmentation_binarization_thresholds()

**Output:** ``eyeball_threshold=0.5 iris_threshold=0.5 pupil_threshold=0.5 eyelashes_threshold=0.5``

As expected all threshold values are set to default ``0.5`` value. Now, let's modify those values to be equal to ``0.1``.

.. code-block:: python

    new_pipeline_conf = {
        "metadata": {"pipeline_name": "iris_pipeline", "iris_version": "1.1.1"},
        "pipeline": [
            {
                "name": "segmentation",
                "algorithm": {"class_name": "iris.MultilabelSegmentation.create_from_hugging_face", "params": {}},
                "inputs": [{"name": "image", "source_node": "input"}],
                "callbacks": None,
            },
    ############################### A NODE, WHICH PARAMETERS WE WANT TO MODIFY ################################
            {
                "name": "segmentation_binarization",
                "algorithm": {"class_name": "iris.MultilabelSegmentationBinarization", "params": {
                    "eyeball_threshold": 0.1,
                    "iris_threshold": 0.1,
                    "pupil_threshold": 0.1,
                    "eyelashes_threshold": 0.1}},
                "inputs": [{"name": "segmentation_map", "source_node": "segmentation"}],
                "callbacks": None,
            },
    ############################################################################################################
            {
                "name": "vectorization",
                "algorithm": {"class_name": "iris.ContouringAlgorithm", "params": {}},
                "inputs": [{"name": "geometry_mask", "source_node": "segmentation_binarization", "index": 0}],
                "callbacks": None,
            },
            {
                "name": "specular_reflection_detection",
                "algorithm": {"class_name": "iris.SpecularReflectionDetection", "params": {}},
                "inputs": [{"name": "ir_image", "source_node": "input"}],
                "callbacks": None,
            },
            {
                "name": "interpolation",
                "algorithm": {"class_name": "iris.ContourInterpolation", "params": {}},
                "inputs": [{"name": "polygons", "source_node": "vectorization"}],
                "callbacks": None,
            },
            {
                "name": "distance_filter",
                "algorithm": {"class_name": "iris.ContourPointNoiseEyeballDistanceFilter", "params": {}},
                "inputs": [
                    {"name": "polygons", "source_node": "interpolation"},
                    {"name": "geometry_mask", "source_node": "segmentation_binarization", "index": 1},
                ],
                "callbacks": None,
            },
            {
                "name": "eye_orientation",
                "algorithm": {"class_name": "iris.MomentOfArea", "params": {}},
                "inputs": [{"name": "geometries", "source_node": "distance_filter"}],
                "callbacks": None,
            },
            {
                "name": "eye_center_estimation",
                "algorithm": {"class_name": "iris.BisectorsMethod", "params": {}},
                "inputs": [{"name": "geometries", "source_node": "distance_filter"}],
                "callbacks": None,
            },
            {
                "name": "smoothing",
                "algorithm": {"class_name": "iris.Smoothing", "params": {}},
                "inputs": [
                    {"name": "polygons", "source_node": "distance_filter"},
                    {"name": "eye_centers", "source_node": "eye_center_estimation"},
                ],
                "callbacks": None,
            },
            {
                "name": "geometry_estimation",
                "algorithm": {
                    "class_name": "iris.FusionExtrapolation",
                    "params": {
                        "circle_extrapolation": {"class_name": "iris.LinearExtrapolation", "params": {"dphi": 0.703125}},
                        "ellipse_fit": {"class_name": "iris.LSQEllipseFitWithRefinement", "params": {"dphi": 0.703125}},
                        "algorithm_switch_std_threshold": 3.5,
                    },
                },
                "inputs": [
                    {"name": "input_polygons", "source_node": "smoothing"},
                    {"name": "eye_center", "source_node": "eye_center_estimation"},
                ],
                "callbacks": None,
            },
            {
                "name": "pupil_to_iris_property_estimation",
                "algorithm": {"class_name": "iris.PupilIrisPropertyCalculator", "params": {}},
                "inputs": [
                    {"name": "geometries", "source_node": "geometry_estimation"},
                    {"name": "eye_centers", "source_node": "eye_center_estimation"},
                ],
                "callbacks": None,
            },
            {
                "name": "offgaze_estimation",
                "algorithm": {"class_name": "iris.EccentricityOffgazeEstimation", "params": {}},
                "inputs": [{"name": "geometries", "source_node": "geometry_estimation"}],
                "callbacks": None,
            },
            {
                "name": "occlusion90_calculator",
                "algorithm": {"class_name": "iris.OcclusionCalculator", "params": {"quantile_angle": 90.0}},
                "inputs": [
                    {"name": "noise_mask", "source_node": "segmentation_binarization", "index": 1},
                    {"name": "extrapolated_polygons", "source_node": "geometry_estimation"},
                    {"name": "eye_orientation", "source_node": "eye_orientation"},
                    {"name": "eye_centers", "source_node": "eye_center_estimation"},
                ],
                "callbacks": None,
            },
            {
                "name": "occlusion30_calculator",
                "algorithm": {"class_name": "iris.OcclusionCalculator", "params": {"quantile_angle": 30.0}},
                "inputs": [
                    {"name": "noise_mask", "source_node": "segmentation_binarization", "index": 1},
                    {"name": "extrapolated_polygons", "source_node": "geometry_estimation"},
                    {"name": "eye_orientation", "source_node": "eye_orientation"},
                    {"name": "eye_centers", "source_node": "eye_center_estimation"},
                ],
                "callbacks": None,
            },
            {
                "name": "noise_masks_aggregation",
                "algorithm": {"class_name": "iris.NoiseMaskUnion", "params": {}},
                "inputs": [
                    {
                        "name": "elements",
                        "source_node": [
                            {"name": "segmentation_binarization", "index": 1},
                            {"name": "specular_reflection_detection"},
                        ],
                    }
                ],
                "callbacks": None,
            },
            {
                "name": "normalization",
                "algorithm": {"class_name": "iris.PerspectiveNormalization", "params": {}},
                "inputs": [
                    {"name": "image", "source_node": "input"},
                    {"name": "noise_mask", "source_node": "noise_masks_aggregation"},
                    {"name": "extrapolated_contours", "source_node": "geometry_estimation"},
                    {"name": "eye_orientation", "source_node": "eye_orientation"},
                ],
                "callbacks": None,
            },
            {
                "name": "filter_bank",
                "algorithm": {
                    "class_name": "iris.ConvFilterBank",
                    "params": {
                        "filters": [
                            {
                                "class_name": "iris.GaborFilter",
                                "params": {
                                    "kernel_size": [41, 21],
                                    "sigma_phi": 7,
                                    "sigma_rho": 6.13,
                                    "theta_degrees": 90.0,
                                    "lambda_phi": 28.0,
                                    "dc_correction": True,
                                    "to_fixpoints": True,
                                },
                            },
                            {
                                "class_name": "iris.GaborFilter",
                                "params": {
                                    "kernel_size": [17, 21],
                                    "sigma_phi": 2,
                                    "sigma_rho": 5.86,
                                    "theta_degrees": 90.0,
                                    "lambda_phi": 8,
                                    "dc_correction": True,
                                    "to_fixpoints": True,
                                },
                            },
                        ],
                        "probe_schemas": [
                            {"class_name": "iris.RegularProbeSchema", "params": {"n_rows": 16, "n_cols": 256}},
                            {"class_name": "iris.RegularProbeSchema", "params": {"n_rows": 16, "n_cols": 256}},
                        ],
                    },
                },
                "inputs": [{"name": "normalization_output", "source_node": "normalization"}],
                "callbacks": None,
            },
            {
                "name": "encoder",
                "algorithm": {"class_name": "iris.IrisEncoder", "params": {}},
                "inputs": [{"name": "response", "source_node": "filter_bank"}],
                "callbacks": None,
            },
            {
                "name": "bounding_box_estimation",
                "algorithm": {"class_name": "iris.IrisBBoxCalculator", "params": {}},
                "inputs": [
                    {"name": "ir_image", "source_node": "input"},
                    {"name": "geometry_polygons", "source_node": "geometry_estimation"},
                ],
                "callbacks": None,
            },
        ],
    }

Reinstantiate ``IRISPipeline`` object.

.. code-block:: python

    iris_pipeline = iris.IRISPipeline(config=new_pipeline_conf)

Print ``iris.MultilabelSegmentationBinarization`` threshold values.

.. code-block:: python

    print_segmentation_binarization_thresholds()

**Output:** ``eyeball_threshold=0.1 iris_threshold=0.1 pupil_threshold=0.1 eyelashes_threshold=0.1``

Perfect! We've just learned how to modify ``IRISPipeline`` algorithms parameters. Now, let's have a look how to modify ``IRISPipeline`` node connections.

2. Configure ``IRISPipeline`` graph.
------------------------------------------------------

As described in previous section to define connection between nodes, we utilize ``inputs`` key within our YAML file or dictionary. Similar to previous tutorial, let's start with instantiating a default ``IRISPipeline`` and then modify "artificially" for demonstration purposes connections between ``distance_filter`` (``iris.ContourPointNoiseEyeballDistanceFilter``), ``smoothing`` (``iris.Smoothing``) and ``geometry_estimation`` (``iris.FusionExtrapolation``) nodes.

By default, ``smoothing`` node, responsible for refinement of vectorized iris and pupil points is taking as an input the output of ``distance_filter`` nodes, which btw is also doing refinement of vectorized iris and pupil points but of course a different one. The output of ``smoothing`` node is later passed to final ``geometry_estimation`` node as an input. Within commented section below user can follow that connection. Now, in this example let's imagine we want to bypass ``smoothing`` node and perform ``geometry_estimation`` based on the output of ``distance_filter`` node while still keeping ``smoothing`` node.

First let's instantiate ``IRISPipeline`` with default configuration and see nodes connected to ``geometry_estimation`` node.

.. code-block:: python

    default_pipeline_conf = {
        "metadata": {"pipeline_name": "iris_pipeline", "iris_version": "1.1.1"},
        "pipeline": [
            {
                "name": "segmentation",
                "algorithm": {"class_name": "iris.MultilabelSegmentation.create_from_hugging_face", "params": {}},
                "inputs": [{"name": "image", "source_node": "input"}],
                "callbacks": None,
            },
            {
                "name": "segmentation_binarization",
                "algorithm": {"class_name": "iris.MultilabelSegmentationBinarization", "params": {
                    "eyeball_threshold": 0.1,
                    "iris_threshold": 0.1,
                    "pupil_threshold": 0.1,
                    "eyelashes_threshold": 0.1}},
                "inputs": [{"name": "segmentation_map", "source_node": "segmentation"}],
                "callbacks": None,
            },
            {
                "name": "vectorization",
                "algorithm": {"class_name": "iris.ContouringAlgorithm", "params": {}},
                "inputs": [{"name": "geometry_mask", "source_node": "segmentation_binarization", "index": 0}],
                "callbacks": None,
            },
            {
                "name": "specular_reflection_detection",
                "algorithm": {"class_name": "iris.SpecularReflectionDetection", "params": {}},
                "inputs": [{"name": "ir_image", "source_node": "input"}],
                "callbacks": None,
            },
            {
                "name": "interpolation",
                "algorithm": {"class_name": "iris.ContourInterpolation", "params": {}},
                "inputs": [{"name": "polygons", "source_node": "vectorization"}],
                "callbacks": None,
            },
    ############################### A NODE, WHICH PARAMETERS WE WANT TO MODIFY ################################
            {
                "name": "distance_filter",
                "algorithm": {"class_name": "iris.ContourPointNoiseEyeballDistanceFilter", "params": {}},
                "inputs": [
                    {"name": "polygons", "source_node": "interpolation"},
                    {"name": "geometry_mask", "source_node": "segmentation_binarization", "index": 1},
                ],
                "callbacks": None,
            },
            {
                "name": "eye_orientation",
                "algorithm": {"class_name": "iris.MomentOfArea", "params": {}},
                "inputs": [{"name": "geometries", "source_node": "distance_filter"}],
                "callbacks": None,
            },
            {
                "name": "eye_center_estimation",
                "algorithm": {"class_name": "iris.BisectorsMethod", "params": {}},
                "inputs": [{"name": "geometries", "source_node": "distance_filter"}],
                "callbacks": None,
            },
            {
                "name": "smoothing",
                "algorithm": {"class_name": "iris.Smoothing", "params": {}},
                "inputs": [
                    {"name": "polygons", "source_node": "distance_filter"},
                    {"name": "eye_centers", "source_node": "eye_center_estimation"},
                ],
                "callbacks": None,
            },
            {
                "name": "geometry_estimation",
                "algorithm": {
                    "class_name": "iris.FusionExtrapolation",
                    "params": {
                        "circle_extrapolation": {"class_name": "iris.LinearExtrapolation", "params": {"dphi": 0.703125}},
                        "ellipse_fit": {"class_name": "iris.LSQEllipseFitWithRefinement", "params": {"dphi": 0.703125}},
                        "algorithm_switch_std_threshold": 3.5,
                    },
                },
                "inputs": [
                    {"name": "input_polygons", "source_node": "smoothing"},
                    {"name": "eye_center", "source_node": "eye_center_estimation"},
                ],
                "callbacks": None,
            },
    ############################################################################################################
            {
                "name": "pupil_to_iris_property_estimation",
                "algorithm": {"class_name": "iris.PupilIrisPropertyCalculator", "params": {}},
                "inputs": [
                    {"name": "geometries", "source_node": "geometry_estimation"},
                    {"name": "eye_centers", "source_node": "eye_center_estimation"},
                ],
                "callbacks": None,
            },
            {
                "name": "offgaze_estimation",
                "algorithm": {"class_name": "iris.EccentricityOffgazeEstimation", "params": {}},
                "inputs": [{"name": "geometries", "source_node": "geometry_estimation"}],
                "callbacks": None,
            },
            {
                "name": "occlusion90_calculator",
                "algorithm": {"class_name": "iris.OcclusionCalculator", "params": {"quantile_angle": 90.0}},
                "inputs": [
                    {"name": "noise_mask", "source_node": "segmentation_binarization", "index": 1},
                    {"name": "extrapolated_polygons", "source_node": "geometry_estimation"},
                    {"name": "eye_orientation", "source_node": "eye_orientation"},
                    {"name": "eye_centers", "source_node": "eye_center_estimation"},
                ],
                "callbacks": None,
            },
            {
                "name": "occlusion30_calculator",
                "algorithm": {"class_name": "iris.OcclusionCalculator", "params": {"quantile_angle": 30.0}},
                "inputs": [
                    {"name": "noise_mask", "source_node": "segmentation_binarization", "index": 1},
                    {"name": "extrapolated_polygons", "source_node": "geometry_estimation"},
                    {"name": "eye_orientation", "source_node": "eye_orientation"},
                    {"name": "eye_centers", "source_node": "eye_center_estimation"},
                ],
                "callbacks": None,
            },
            {
                "name": "noise_masks_aggregation",
                "algorithm": {"class_name": "iris.NoiseMaskUnion", "params": {}},
                "inputs": [
                    {
                        "name": "elements",
                        "source_node": [
                            {"name": "segmentation_binarization", "index": 1},
                            {"name": "specular_reflection_detection"},
                        ],
                    }
                ],
                "callbacks": None,
            },
            {
                "name": "normalization",
                "algorithm": {"class_name": "iris.PerspectiveNormalization", "params": {}},
                "inputs": [
                    {"name": "image", "source_node": "input"},
                    {"name": "noise_mask", "source_node": "noise_masks_aggregation"},
                    {"name": "extrapolated_contours", "source_node": "geometry_estimation"},
                    {"name": "eye_orientation", "source_node": "eye_orientation"},
                ],
                "callbacks": None,
            },
            {
                "name": "filter_bank",
                "algorithm": {
                    "class_name": "iris.ConvFilterBank",
                    "params": {
                        "filters": [
                            {
                                "class_name": "iris.GaborFilter",
                                "params": {
                                    "kernel_size": [41, 21],
                                    "sigma_phi": 7,
                                    "sigma_rho": 6.13,
                                    "theta_degrees": 90.0,
                                    "lambda_phi": 28.0,
                                    "dc_correction": True,
                                    "to_fixpoints": True,
                                },
                            },
                            {
                                "class_name": "iris.GaborFilter",
                                "params": {
                                    "kernel_size": [17, 21],
                                    "sigma_phi": 2,
                                    "sigma_rho": 5.86,
                                    "theta_degrees": 90.0,
                                    "lambda_phi": 8,
                                    "dc_correction": True,
                                    "to_fixpoints": True,
                                },
                            },
                        ],
                        "probe_schemas": [
                            {"class_name": "iris.RegularProbeSchema", "params": {"n_rows": 16, "n_cols": 256}},
                            {"class_name": "iris.RegularProbeSchema", "params": {"n_rows": 16, "n_cols": 256}},
                        ],
                    },
                },
                "inputs": [{"name": "normalization_output", "source_node": "normalization"}],
                "callbacks": None,
            },
            {
                "name": "encoder",
                "algorithm": {"class_name": "iris.IrisEncoder", "params": {}},
                "inputs": [{"name": "response", "source_node": "filter_bank"}],
                "callbacks": None,
            },
            {
                "name": "bounding_box_estimation",
                "algorithm": {"class_name": "iris.IrisBBoxCalculator", "params": {}},
                "inputs": [
                    {"name": "ir_image", "source_node": "input"},
                    {"name": "geometry_polygons", "source_node": "geometry_estimation"},
                ],
                "callbacks": None,
            },
        ],
    }

Instantiate ``IRISPipeline`` object.

.. code-block:: python

    iris_pipeline = iris.IRISPipeline(config=default_pipeline_conf)

Print ``geometry_estimation`` input nodes name.

.. code-block:: python

    def print_geometry_estimation_inputs():
        geometry_estimation_node = [node for node in iris_pipeline.params.pipeline if node.name == "geometry_estimation"]

        assert len(geometry_estimation_node) == 1

        geometry_estimation_node = geometry_estimation_node[0]
        print(geometry_estimation_node.inputs)

    print_geometry_estimation_inputs()

**Output:** ``[PipelineInput(name='input_polygons', index=None, source_node='smoothing'), PipelineInput(name='eye_center', index=None, source_node='eye_center_estimation')]``

As expected, ``input_polygons`` argument of the ``run`` method is taken from the ``smoothing`` output. Let's modify it to described before behaviour - ``input_polygons`` argument of the ``run`` method is take from the ``distance_filter`` output.

.. code-block:: python

    new_pipeline_conf = {
        "metadata": {"pipeline_name": "iris_pipeline", "iris_version": "1.1.1"},
        "pipeline": [
            {
                "name": "segmentation",
                "algorithm": {"class_name": "iris.MultilabelSegmentation.create_from_hugging_face", "params": {}},
                "inputs": [{"name": "image", "source_node": "input"}],
                "callbacks": None,
            },
            {
                "name": "segmentation_binarization",
                "algorithm": {"class_name": "iris.MultilabelSegmentationBinarization", "params": {
                    "eyeball_threshold": 0.1,
                    "iris_threshold": 0.1,
                    "pupil_threshold": 0.1,
                    "eyelashes_threshold": 0.1}},
                "inputs": [{"name": "segmentation_map", "source_node": "segmentation"}],
                "callbacks": None,
            },
            {
                "name": "vectorization",
                "algorithm": {"class_name": "iris.ContouringAlgorithm", "params": {}},
                "inputs": [{"name": "geometry_mask", "source_node": "segmentation_binarization", "index": 0}],
                "callbacks": None,
            },
            {
                "name": "specular_reflection_detection",
                "algorithm": {"class_name": "iris.SpecularReflectionDetection", "params": {}},
                "inputs": [{"name": "ir_image", "source_node": "input"}],
                "callbacks": None,
            },
            {
                "name": "interpolation",
                "algorithm": {"class_name": "iris.ContourInterpolation", "params": {}},
                "inputs": [{"name": "polygons", "source_node": "vectorization"}],
                "callbacks": None,
            },
    ############################### A NODE, WHICH PARAMETERS WE WANT TO MODIFY ################################
            {
                "name": "distance_filter",
                "algorithm": {"class_name": "iris.ContourPointNoiseEyeballDistanceFilter", "params": {}},
                "inputs": [
                    {"name": "polygons", "source_node": "interpolation"},
                    {"name": "geometry_mask", "source_node": "segmentation_binarization", "index": 1},
                ],
                "callbacks": None,
            },
            {
                "name": "eye_orientation",
                "algorithm": {"class_name": "iris.MomentOfArea", "params": {}},
                "inputs": [{"name": "geometries", "source_node": "distance_filter"}],
                "callbacks": None,
            },
            {
                "name": "eye_center_estimation",
                "algorithm": {"class_name": "iris.BisectorsMethod", "params": {}},
                "inputs": [{"name": "geometries", "source_node": "distance_filter"}],
                "callbacks": None,
            },
            {
                "name": "smoothing",
                "algorithm": {"class_name": "iris.Smoothing", "params": {}},
                "inputs": [
                    {"name": "polygons", "source_node": "distance_filter"},
                    {"name": "eye_centers", "source_node": "eye_center_estimation"},
                ],
                "callbacks": None,
            },
            {
                "name": "geometry_estimation",
                "algorithm": {
                    "class_name": "iris.FusionExtrapolation",
                    "params": {
                        "circle_extrapolation": {"class_name": "iris.LinearExtrapolation", "params": {"dphi": 0.703125}},
                        "ellipse_fit": {"class_name": "iris.LSQEllipseFitWithRefinement", "params": {"dphi": 0.703125}},
                        "algorithm_switch_std_threshold": 3.5,
                    },
                },
                "inputs": [
                    {"name": "input_polygons", "source_node": "distance_filter"},
                    {"name": "eye_center", "source_node": "eye_center_estimation"},
                ],
                "callbacks": None,
            },
    ############################################################################################################
            {
                "name": "pupil_to_iris_property_estimation",
                "algorithm": {"class_name": "iris.PupilIrisPropertyCalculator", "params": {}},
                "inputs": [
                    {"name": "geometries", "source_node": "geometry_estimation"},
                    {"name": "eye_centers", "source_node": "eye_center_estimation"},
                ],
                "callbacks": None,
            },
            {
                "name": "offgaze_estimation",
                "algorithm": {"class_name": "iris.EccentricityOffgazeEstimation", "params": {}},
                "inputs": [{"name": "geometries", "source_node": "geometry_estimation"}],
                "callbacks": None,
            },
            {
                "name": "occlusion90_calculator",
                "algorithm": {"class_name": "iris.OcclusionCalculator", "params": {"quantile_angle": 90.0}},
                "inputs": [
                    {"name": "noise_mask", "source_node": "segmentation_binarization", "index": 1},
                    {"name": "extrapolated_polygons", "source_node": "geometry_estimation"},
                    {"name": "eye_orientation", "source_node": "eye_orientation"},
                    {"name": "eye_centers", "source_node": "eye_center_estimation"},
                ],
                "callbacks": None,
            },
            {
                "name": "occlusion30_calculator",
                "algorithm": {"class_name": "iris.OcclusionCalculator", "params": {"quantile_angle": 30.0}},
                "inputs": [
                    {"name": "noise_mask", "source_node": "segmentation_binarization", "index": 1},
                    {"name": "extrapolated_polygons", "source_node": "geometry_estimation"},
                    {"name": "eye_orientation", "source_node": "eye_orientation"},
                    {"name": "eye_centers", "source_node": "eye_center_estimation"},
                ],
                "callbacks": None,
            },
            {
                "name": "noise_masks_aggregation",
                "algorithm": {"class_name": "iris.NoiseMaskUnion", "params": {}},
                "inputs": [
                    {
                        "name": "elements",
                        "source_node": [
                            {"name": "segmentation_binarization", "index": 1},
                            {"name": "specular_reflection_detection"},
                        ],
                    }
                ],
                "callbacks": None,
            },
            {
                "name": "normalization",
                "algorithm": {"class_name": "iris.PerspectiveNormalization", "params": {}},
                "inputs": [
                    {"name": "image", "source_node": "input"},
                    {"name": "noise_mask", "source_node": "noise_masks_aggregation"},
                    {"name": "extrapolated_contours", "source_node": "geometry_estimation"},
                    {"name": "eye_orientation", "source_node": "eye_orientation"},
                ],
                "callbacks": None,
            },
            {
                "name": "filter_bank",
                "algorithm": {
                    "class_name": "iris.ConvFilterBank",
                    "params": {
                        "filters": [
                            {
                                "class_name": "iris.GaborFilter",
                                "params": {
                                    "kernel_size": [41, 21],
                                    "sigma_phi": 7,
                                    "sigma_rho": 6.13,
                                    "theta_degrees": 90.0,
                                    "lambda_phi": 28.0,
                                    "dc_correction": True,
                                    "to_fixpoints": True,
                                },
                            },
                            {
                                "class_name": "iris.GaborFilter",
                                "params": {
                                    "kernel_size": [17, 21],
                                    "sigma_phi": 2,
                                    "sigma_rho": 5.86,
                                    "theta_degrees": 90.0,
                                    "lambda_phi": 8,
                                    "dc_correction": True,
                                    "to_fixpoints": True,
                                },
                            },
                        ],
                        "probe_schemas": [
                            {"class_name": "iris.RegularProbeSchema", "params": {"n_rows": 16, "n_cols": 256}},
                            {"class_name": "iris.RegularProbeSchema", "params": {"n_rows": 16, "n_cols": 256}},
                        ],
                    },
                },
                "inputs": [{"name": "normalization_output", "source_node": "normalization"}],
                "callbacks": None,
            },
            {
                "name": "encoder",
                "algorithm": {"class_name": "iris.IrisEncoder", "params": {}},
                "inputs": [{"name": "response", "source_node": "filter_bank"}],
                "callbacks": None,
            },
            {
                "name": "bounding_box_estimation",
                "algorithm": {"class_name": "iris.IrisBBoxCalculator", "params": {}},
                "inputs": [
                    {"name": "ir_image", "source_node": "input"},
                    {"name": "geometry_polygons", "source_node": "geometry_estimation"},
                ],
                "callbacks": None,
            },
        ],
    }

Reinstantiate ``IRISPipeline`` object.

.. code-block:: python

    iris_pipeline = iris.IRISPipeline(config=new_pipeline_conf)

Print ``geometry_estimation`` input nodes name.

.. code-block:: python

    print_geometry_estimation_inputs()

**Output:** ``[PipelineInput(name='input_polygons', index=None, source_node='distance_filter'), PipelineInput(name='eye_center', index=None, source_node='eye_center_estimation')]``

Perfect! Now, we can see that ``geometry_estimation`` will use the output of ``distance_filter`` node as an input. Last but not least, before concluding this tutorial, we have to learn how to implement our own custom nodes that can be plugged to ``IRISPipeline``.

3. Implement your own node with ``Algorithm`` class and introduce them into ``IRISPipeline`` graph.
------------------------------------------------------------------------------------------------------------

The ``Algorithm`` class is an abstract class that is a base class for every node and ``IRISPipeline`` in the ``iris`` packages. It's defined as follow:

.. code-block:: python

    class Algorithm(abc.ABC):
        """Base class of every node of the iris recognition pipeline."""

        class Parameters(ImmutableModel):
            """Default parameters."""

            pass

        __parameters_type__ = Parameters

        def __init__(self, **kwargs: Any) -> None:
            """Init function."""
            self._callbacks: List[Callback] = []

            if "callbacks" in kwargs.keys():
                self._callbacks = deepcopy(kwargs["callbacks"])
                del kwargs["callbacks"]

            self.params = self.__parameters_type__(**kwargs)

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            """Make an object a functor.

            Returns:
                Any: Object specified by an interface.
            """
            return self.execute(*args, **kwargs)

        def execute(self, *args: Any, **kwargs: Any) -> Any:
            """Execute method and wrapped with hooks if such are specified.

            Returns:
                Any: Object specified by an interface.
            """
            for callback_func in self._callbacks:
                callback_func.on_execute_start(*args, **kwargs)

            result = self.run(*args, **kwargs)

            for callback_func in self._callbacks:
                callback_func.on_execute_end(result)

            return result

        def run(self, *args: Any, **kwargs: Any) -> Any:
            """Implement method design pattern. Not overwritten by subclass will raise an error.

            Raises:
                NotImplementedError: Raised if subclass doesn't implement `run` method.

            Returns:
                Any: Return value by concrete implementation of the `run` method.
            """
            raise NotImplementedError(f"{self.__class__.__name__}.run method not implemented!")

There are 3 important things to note that have direct implications on how user have to implement custom ``Algorithm``:

* The ``run`` method - If we implement our own custom ``Algorithm`` we have to make sure that ``run`` method is implemented. Other then that, already mentioned callbacks.
* The ``__parameters_type__`` variable - In our code base, we use ``pydantic`` package to perform validation of ``Algorithm`` ``__init__`` parameters. To simplify and hide behind the screen those mechanisms, we introduced this variable.
* The ``callbacks`` special key that can be provided in the ``__init__`` method. As already mentioned before, if we want to turn on in our ``Algorithm`` callbacks mechanisms, we have to specify special - ``callbacks`` - parameter in that ``Algorithm`` ``__init__`` method.

In this section, we won't provide examples since there are plenty of them within the ``iris`` package. Plus, we also want to encourage you to explore the ``iris`` package by yourself. Therefore, for examples of concrete ``Algorithm`` implementations, please check ``iris.nodes`` submodule of the ``iris`` package.

**Thank you for making it to the end of this tutorial!**
