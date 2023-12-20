Installation
================================

Installation is as simple as running ``pip install`` with specifying ``IRIS_ENV`` installation global flag (``IRIS_ENV`` flag may be skipped if ``iris`` is installed from PyPl server but this option is only available when ``iris`` is installed on local machine). The ``IRIS_ENV`` flag is used to indicate an "environment" in which package is meant to work. Possible options are:

#. ``SERVER`` - For installing ``iris`` package with dependencies required for running an inference on a local machines.

.. code:: bash

    # On a local machine
    pip install open-iris
    # or directly from GitHub
    IRIS_ENV=SERVER pip install git+https://github.com/worldcoin/open-iris.git

#. ``ORB`` - For installing ``iris`` package with dependencies required for running an inference on the Orb.

.. code:: bash

    # On the Orb
    IRIS_ENV=ORB pip install git+https://github.com/worldcoin/open-iris.git

#. ``DEV`` - For installing iris package together with packages necessary for development of ``iris`` package.

.. code:: bash

    # For development
    IRIS_ENV=DEV pip install git+https://github.com/worldcoin/open-iris.git

After successfully installing ``iris``, verify your installation by attempting to import.

.. code:: bash

    python3 -c "import iris; print(iris.__version__)"
