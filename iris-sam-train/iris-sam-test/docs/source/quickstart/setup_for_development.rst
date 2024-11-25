Setup for development
================================

A ``conda`` environment simplifies the setup process for developing on the ``iris`` package. This ``conda`` environment ensures a seamless and consistent setup for contributors, reducing the complexity of dependency management. By utilizing ``conda``, developers can easily replicate the development environment across different systems, minimizing potential setup obstacles. This approach aims to make it straightforward for anyone interested in contributing to quickly set up and engage in the development of ``iris`` package.

.. code-block:: bash

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
