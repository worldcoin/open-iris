#################################################################################
#  The script reads and checks if release's version matches package's version.  #
#               Used in `check-pre-release` Github Action.                      #
#                       Author: Worldcoin AI                                    #
#################################################################################

import argparse
import importlib


class ReleaseVersionMismatchError(Exception):
    """Release version mismatch error class."""

    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--release", help="Desired version tag", required=True)

    args = parser.parse_args()

    module_name = "iris"
    module = importlib.import_module(module_name)

    if f"v{module.__version__}" != args.release:
        raise ReleaseVersionMismatchError(
            f"Check your release tag and module version {module_name}. {module.__version__} != {args.release}"
        )
