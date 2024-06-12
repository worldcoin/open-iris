______________________________________________________________________
<div align="center">

<p align="center">
  <a href="https://worldcoin.org/"><img src="https://github.com/worldcoin/open-iris/blob/main/docs/source/images/logos/wld.png?raw=true" width=150px></img></a>
</p>

# **_IRIS: Iris Recognition Inference System_**

<a href="https://worldcoin.github.io/open-iris/">Package documentation</a> •
<a href="https://colab.research.google.com/github/worldcoin/open-iris/blob/main/colab/GettingStarted.ipynb">_Getting started_ tutorial</a> •
<a href="https://huggingface.co/Worldcoin/iris-semantic-segmentation/tree/main">Hugging Face repo</a> •
<a href="https://worldcoin.org/blog/engineering/iris-recognition-inference-system">IRIS blog post</a>

[![python](https://img.shields.io/badge/-Python_3.8_%7C_3.9_%7C_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![pydocstyle](https://img.shields.io/badge/pydocstyle-enabled-AD4CD3)](http://www.pydocstyle.org/en/stable/) <br />
[![ci-testing](https://github.com/worldcoin/open-iris/actions/workflows/ci-testing.yml/badge.svg?branch=main&event=push)](https://github.com/worldcoin/open-iris/actions/workflows/ci-testing.yml)
[![release-version](https://github.com/worldcoin/open-iris/actions/workflows/check-release-version.yml/badge.svg)](https://github.com/worldcoin/open-iris/actions/workflows/check-release-version.yml)

______________________________________________________________________

</div>

## Table of contents

- [About](#about)
- [Quickstart](#quickstart)
  - [Installation](#installation)
  - [Setup for development](#setup-for-development)
  - [Running inference](#Running-inference)
- [Disclaimer](#disclaimer)
- [Project structure](#project-structure)
- [Example notebooks](#example-notebooks)
- [Documentation](#documentation)
- [Issues, pull requests and feature requests](#issues-pull-requests-and-feature-requests)
- [Citation](#citation)
- [License](#license)
- [Resources](#resources)

## About

Welcome to Worldcoin's Iris Recognition Inference System (IRIS) project, an advanced iris recognition pipeline designed for robust and secure biometric verification. This project leverages state-of-the-art computer vision and machine learning techniques to provide accurate and efficient iris recognition system.

Iris recognition is a powerful biometric technology that identifies individuals based on the unique patterns within the iris of the eye. IRIS package aims to make iris recognition accessible and enable further advancement in the field.

Project features highlights are:
- **Large-Scale Verification**: Capable of verifying uniqueness among billions of users.
- **High-Performance Iris Segmentation**: Accurate segmentation of iris regions for precise feature extraction.
- **Scalable Matching Algorithm**: Robust matching algorithm designed for scalability without compromising accuracy.
- **User-Friendly Integration**: Simple integration into applications that demand seamless biometric verification.

High-level iris recognition pipeline steps overview:
1. **Iris Image Input**: Provide an iris image for verification.
2. **Iris Segmentation**: Identify and isolate the iris region within the image.
3. **Feature Extraction**: Extract unique features from the iris to create a template.
4. **Scalable Matching**: Efficiently compare extracted features for large-scale uniqueness verification.
5. **Result**: Receive the verification result with a confidence score, enabling secure and scalable authentication.

The Worldcoin system utilizes iris recognition algorithm for verifying uniqueness in a challenging environment, involving billions of individuals. This entails a detailed exploration of the Worldcoin biometric pipeline, a system that confirms uniqueness through the encoding of iris texture into an iris code.

More detailed pipeline overview can be found in our [blog post](https://worldcoin.org/blog/engineering/iris-recognition-inference-system) dedicated to IRIS project.


## Disclaimer

_The Iris Recognition Inference System (IRIS) software repository is owned and maintained by the Worldcoin Foundation, the steward of the Worldcoin protocol; the repository is not affiliated with any other project or service provider_


## Quickstart

### Installation

Installation is as simple as running `pip install` with specifying `IRIS_ENV` installation global flag (`IRIS_ENV` flag may be skipped if `iris` is installed from PyPl server but this option is only available when `iris` is installed on local machine). The `IRIS_ENV` flag is used to indicate an "environment" in which package is meant to work. Possible options are:
1. `SERVER` - For installing `iris` package with dependencies required for running an inference on a local machines.
```bash
# On a local machine
pip install open-iris
# or directly from GitHub
IRIS_ENV=SERVER pip install git+https://github.com/worldcoin/open-iris.git
```
2. `ORB` - For installing `iris` package with dependencies required for running an inference on the Orb.
```bash
# On the Orb
IRIS_ENV=ORB pip install git+https://github.com/worldcoin/open-iris.git
```
3. `DEV` - For installing iris package together with packages necessary for development of `iris` package.
```bash
# For development
IRIS_ENV=DEV pip install git+https://github.com/worldcoin/open-iris.git
```

After successfully installing `iris`, verify your installation by attempting to import.
```bash
python3 -c "import iris; print(iris.__version__)"
```

### Setup for development

A `conda` environment simplifies the setup process for developing on the `iris` package. This `conda` environment ensures a seamless and consistent setup for contributors, reducing the complexity of dependency management. By utilizing `conda`, developers can easily replicate the development environment across different systems, minimizing potential setup obstacles. This approach aims to make it straightforward for anyone interested in contributing to quickly set up and engage in the development of `iris` package.

```bash
# Clone the iris repo
git clone https://github.com/worldcoin/open-iris

# Go to the repo directory
cd open-iris

# Create and activate conda environment
IRIS_ENV=DEV conda env create -f ./conda/environment_dev.yml
conda activate iris_dev

# (Optional, but recommended) Install git hooks to preserve code format consistency
pre-commit install
nb-clean add-filter --remove-empty-cells
```

### Running inference

A simple inference run can be achived by running source code below.

```python
import cv2
import iris

# 1. Create IRISPipeline object
iris_pipeline = iris.IRISPipeline()

# 2. Load IR image of an eye
img_pixels = cv2.imread("/path/to/ir/image", cv2.IMREAD_GRAYSCALE)

# 3. Perform inference
# Options for the `eye_side` argument are: ["left", "right"]
output = iris_pipeline(img_data=img_pixels, eye_side="left")
```

To fully explore and understand the extensive capabilities of the iris package, visit the [Example notebooks](#example-notebooks) section. Here, you'll find a collection of Jupyter Notebooks that serve as valuable resources, offering practical guides and real-world examples to provide a comprehensive insight into the rich functionalities and potential applications of the `iris` package.

## Project structure

The `iris` package features a structured design with modular components, enhancing code organization and scalability.

| **Module** | **Description** |
|----------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| [`iris`](https://worldcoin.github.io/open-iris/) | an iris recognition package |
| [`iris.callbacks`](https://worldcoin.github.io/open-iris/) | a module that implements callbacks used to customize on execute start and end behaviours of node or pipeline call |
| [`iris.io`](https://worldcoin.github.io/open-iris/) | a module that contains dataclasses and errors that flow through iris recognition pipeline when called |
| [`iris.nodes`](https://worldcoin.github.io/open-iris/) | a module that contains implementation of iris recognition pipeline nodes |
| [`iris.orchestration`](https://worldcoin.github.io/open-iris/) | a module that contains iris recognition pipeline's orchestration support mechanisms |
| [`iris.pipelines`](https://worldcoin.github.io/open-iris/) | a module that contains implementation of iris recognition pipelines |
| [`iris.utils`](https://worldcoin.github.io/open-iris/) | a module that contains utilities used throughout the code base and modules useful for outputs analysis |

## Example notebooks

The Jupyter Notebooks provided present practical guides and real-world instances to demonstrate the complete capabilities of the `iris` package.

1. **Getting started** [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/worldcoin/open-iris/blob/main/colab/GettingStarted.ipynb)
2. **Configuring custom pipeline** [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/worldcoin/open-iris/blob/main/colab/ConfiguringCustomPipeline.ipynb)
3. **Matching entities** [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/worldcoin/open-iris/blob/main/colab/MatchingEntities.ipynb)

## Documentation

For detailed documentation, including installation instructions, usage guidelines, and configuration options, please refer to the IRIS project [documentation](https://worldcoin.github.io/open-iris/).

## Issues, pull requests and feature requests

If you have any question or you found a bug or you feel like some feature is missing, please don't hesitate to file a new [issue](https://github.com/worldcoin/open-iris/issues), discussion or [PR](https://github.com/worldcoin/open-iris/pulls) with respective title and description.
Any suggestion for potential project improvements are and will always be welcome!

## Contact

If you have any questions, suggestions, or feedback, feel free to reach out to us:

- Email: [iris@toolsforhumanity.com](mailto:iris@toolsforhumanity.com)
- GitHub Issues: [Open an issue](https://github.com/worldcoin/open-iris/issues)
- Contributors: Feel free to reach out to any project [contributor](https://github.com/worldcoin/open-iris/graphs/contributors) directly!

## Citation
```BibTeX
@misc{wldiris,
  author =       {Worldcoin AI},
  title =        {IRIS: Iris Recognition Inference System of the Worldcoin project},
  year =         {2023},
  url =          {https://github.com/worldcoin/open-iris}
}
```

## License

This project is licensed under the [MIT license](https://github.com/worldcoin/open-iris/blob/main/LICENSE).

## Resources

1. [_"Iris Recognition Inference System"_](https://worldcoin.org/blog/engineering/iris-recognition-inference-system)
2. [_"Iris feature extraction with 2D Gabor wavelets"_](https://worldcoin.org/blog/engineering/iris-feature-extraction)
3. [_"How iris recognition works"_](https://ieeexplore.ieee.org/document/1262028)
