.. image:: /images/logos/wld.png
   :alt: Worldcoin AI logo
   :align: center
   :scale: 65%

`Iris Recognition Inference System`
-------------------------------------------

Welcome to Worldcoin's Iris Recognition Inference System (IRIS) project, an advanced iris recognition pipeline designed for robust and secure biometric verification. This project leverages state-of-the-art computer vision and machine learning techniques to provide accurate and efficient iris recognition system.

Iris recognition is a powerful biometric technology that identifies individuals based on the unique patterns within the iris of the eye. IRIS package aims to make iris recognition accessible and enable further advancement in the field.

Project features highlights are:

- **Large-Scale Verification**: Capable of verifying uniqueness among billions of users.
- **High-Performance Iris Segmentation**: Accurate segmentation of iris regions for precise feature extraction.
- **Scalable Matching Algorithm**: Robust matching algorithm designed for scalability without compromising accuracy.
- **User-Friendly Integration**: Simple integration into applications that demand seamless biometric verification.

High-level iris recognition pipeline steps overview:

#. **Iris Image Input**: Provide an iris image for verification.
#. **Iris Segmentation**: Identify and isolate the iris region within the image.
#. **Feature Extraction**: Extract unique features from the iris to create a template.
#. **Scalable Matching**: Efficiently compare extracted features for large-scale uniqueness verification.
#. **Result**: Receive the verification result with a confidence score, enabling secure and scalable authentication.

The Worldcoin system utilizes iris recognition algorithm for verifying uniqueness in a challenging environment, involving billions of individuals. This entails a detailed exploration of the Worldcoin biometric pipeline, a system that confirms uniqueness through the encoding of iris texture into an iris code.

More detailed pipeline overview can be found in our `blog post <https://worldcoin.org/blog/engineering/iris-recognition-inference-system>`_ dedicated to IRIS project.

**Disclaimer**

*The Iris Recognition Inference System (IRIS) software repository is owned and maintained by the Worldcoin Foundation, the steward of the Worldcoin protocol; the repository is not affiliated with any other project or service provider*

.. toctree::
   :hidden:
   :caption: Quickstart

   quickstart/installation
   quickstart/setup_for_development
   quickstart/running_inference

.. toctree::
   :hidden:
   :caption: Examples

   examples/getting_started
   examples/custom_pipeline
   examples/matching_entities

.. toctree::
   :hidden:
   :caption: Issues, pull requests and feature requests

   issues_note

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: API Reference

   _code_subpages/modules
