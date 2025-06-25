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
- [Performance & Security](#performance--security)
- [License](#license)

## About

**IRIS** is Worldcoin origin open-source Iris Recognition Inference System, designed for robust, scalable, and secure biometric identification. This fork of it now uses a hash-based matching approach, generating a 40-bit unique identifier for each iris, enabling O(1) lookup and cryptographic security at global scale.

## Features
- **Hash-Based Matching**: Each iris template is converted to a 40-bit unique ID using SHA-256, enabling exact, fast, and secure matching.
- **O(1) Lookup**: Matching is performed by comparing 40-bit integers, making the system highly scalable for billions of users.
- **Cryptographic Security**: Uses SHA-256 for template hashing, ensuring strong privacy and resistance to tampering.
- **Modular Pipeline**: Includes segmentation, feature extraction, and template generation.

## Installation

```
git clone https://github.com/irhdab/open-iris.git
```

## Performance & Security
- **Storage**: Each iris is represented by a 40-bit (5 byte) unique ID, reducing storage by over 1,000x compared to traditional templates.
- **Speed**: Matching is O(1) (integer comparison), suitable for massive-scale deployments.
- **Security**: SHA-256 hashing ensures cryptographic strength and privacy.

## License

This project is licensed under the [MIT license](LICENSE).

---
