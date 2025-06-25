______________________________________________________________________
<div align="center">

<p align="center">
  <a href="https://worldcoin.org/"><img src="https://github.com/worldcoin/open-iris/blob/main/docs/source/images/logos/wld.png?raw=true" width=150px></img></a>
</p>

# **_IRIS: Iris Recognition Inference System_**

<a href="https://worldcoin.github.io/open-iris/">Documentation</a> •
<a href="https://colab.research.google.com/github/worldcoin/open-iris/blob/main/colab/GettingStarted.ipynb">Quickstart Notebook</a> •
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

## Table of Contents
- [About](#about)
- [Features](#features)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [API Overview](#api-overview)
- [Performance & Security](#performance--security)
- [Contributing](#contributing)
- [License](#license)

## About

**IRIS** is Worldcoin's open-source Iris Recognition Inference System, designed for robust, scalable, and secure biometric identification. The system now uses a hash-based matching approach, generating a 40-bit unique identifier for each iris, enabling O(1) lookup and cryptographic security at global scale.

## Features
- **Hash-Based Matching**: Each iris template is converted to a 40-bit unique ID using SHA-256, enabling exact, fast, and secure matching.
- **O(1) Lookup**: Matching is performed by comparing 40-bit integers, making the system highly scalable for billions of users.
- **Cryptographic Security**: Uses SHA-256 for template hashing, ensuring strong privacy and resistance to tampering.
- **Modular Pipeline**: Includes segmentation, feature extraction, and template generation.
- **Easy Integration**: Simple Python API for embedding iris recognition in your applications.

## Installation

Install from PyPI:
```bash
pip install open-iris
```
Or for the latest version:
```bash
IRIS_ENV=SERVER pip install git+https://github.com/worldcoin/open-iris.git
```

## Quickstart

```python
import cv2
import iris

# 1. Create IRISPipeline object
iris_pipeline = iris.IRISPipeline()

# 2. Load IR image of an eye
eyepixels = cv2.imread("/path/to/ir/image", cv2.IMREAD_GRAYSCALE)

# 3. Generate iris template
output = iris_pipeline(img_data=eyepixels, eye_side="left")
template = output['iris_template']

# 4. Generate 40-bit unique identifier
matcher = iris.HashBasedMatcher()
unique_id = matcher.generate_unique_id(template)
print(f"Iris unique ID: {unique_id}")

# 5. Compare two templates (returns 0.0 for exact match, 1.0 for no match)
# template2 = ... (another iris template)
# match_score = matcher.run(template, template2)
```

## API Overview

- **IRISPipeline**: Main pipeline for segmentation, feature extraction, and template generation.
- **IrisTemplate**: Data class holding the iris code and mask. Use `generate_unique_id()` to get the 40-bit ID.
- **HashBasedMatcher**: Main matcher class. Use `generate_unique_id(template)` or `run(template1, template2)`.

## Performance & Security
- **Storage**: Each iris is represented by a 40-bit (5 byte) unique ID, reducing storage by over 1,000x compared to traditional templates.
- **Speed**: Matching is O(1) (integer comparison), suitable for massive-scale deployments.
- **Security**: SHA-256 hashing ensures cryptographic strength and privacy.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) (if available) or open an issue/PR.

## License

This project is licensed under the [MIT license](LICENSE).

---

**Contact:** [iris@toolsforhumanity.com](mailto:iris@toolsforhumanity.com)

For more details, see the [documentation](https://worldcoin.github.io/open-iris/) and [example notebooks](https://colab.research.google.com/github/worldcoin/open-iris/blob/main/colab/GettingStarted.ipynb).
