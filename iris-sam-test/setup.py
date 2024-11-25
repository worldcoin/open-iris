import os
import runpy
from typing import List

import setuptools


class IRISInstallationError(Exception):
    """The `iris` package installation error class."""

    pass


def load_requirements(reqs_filename: str) -> List[str]:
    """Load requirements from a txt file ./requirements/{reqs_filename.txt}.

    Args:
        reqs_filename (str): The requirements.txt file name.

    Returns:
        List[str]: A list of requirements.
    """
    reqs_dirpath = os.path.join(os.path.dirname(__file__), "requirements")
    reqs_filepath = os.path.join(reqs_dirpath, reqs_filename)

    if not os.path.isfile(reqs_filepath):
        raise IRISInstallationError(f"Invalid req_filename: {reqs_filename} isn't a file.")

    if not reqs_filename.endswith(".txt"):
        raise IRISInstallationError(f"Invalid req_filename: {reqs_filename} doesn't end with .txt")

    requirements = [line.strip() for line in open(reqs_filepath).readlines()]

    return requirements


def load_description() -> str:
    """Load description from README.md file.

    Returns:
        str: A package description.
    """
    readme_filepath = os.path.join(os.path.dirname(__file__), "README.md")

    if not os.path.exists(readme_filepath):
        return ""

    with open(readme_filepath, "r", encoding="UTF-8") as fh:
        long_description = fh.read()

    return long_description


if __name__ == "__main__":
    IRIS_ENV_STRING_KEY = "IRIS_ENV"
    AVAILABLE_IRIS_ENV = ["SERVER", "ORB", "DEV"]

    if IRIS_ENV_STRING_KEY not in os.environ:
        raise IRISInstallationError(
            f"Environment variable {IRIS_ENV_STRING_KEY} not specified. Possible options are: {AVAILABLE_IRIS_ENV}."
        )

    if os.environ[IRIS_ENV_STRING_KEY].strip() not in AVAILABLE_IRIS_ENV:
        raise IRISInstallationError(
            f"Invalid environment variable {IRIS_ENV_STRING_KEY} specified. Possible options are: {AVAILABLE_IRIS_ENV}."
        )

    base_requirements = load_requirements(reqs_filename="base.txt")
    server_requirements = [*base_requirements, *load_requirements(reqs_filename="server.txt")]
    orb_requirements = [*base_requirements, *load_requirements(reqs_filename="orb.txt")]
    dev_requirements = [*server_requirements, *load_requirements(reqs_filename="dev.txt")]

    iris_packages = setuptools.find_packages()

    IRIS_ENV_INSTALLATION_CONFIGS = {
        "SERVER": {"install_requires": server_requirements},
        "ORB": {"install_requires": orb_requirements},
        "DEV": {"install_requires": dev_requirements},
    }

    setuptools.setup(
        name="open-iris",
        package_dir={"": "src"},
        packages=setuptools.find_packages("src"),
        version=runpy.run_path("src/iris/_version.py")["__version__"],
        author="Worldcoin AI",
        author_email="ai@worldcoin.org",
        description="IRIS: Iris Recognition Inference System of the Worldcoin project.",
        long_description=load_description(),
        long_description_content_type="text/markdown",
        url="https://github.com/worldcoin/open-iris",
        classifiers=[
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Operating System :: OS Independent",
        ],
        python_requires=">=3.8",
        license="MIT",
        keywords=["biometrics", "iris recognition"],
        include_package_data=True,
        **IRIS_ENV_INSTALLATION_CONFIGS[os.environ[IRIS_ENV_STRING_KEY]],
    )
