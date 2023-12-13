import os
import sys

script_path = os.path.abspath(__file__)
tests_dir = os.path.dirname(script_path)
project_root = os.path.dirname(tests_dir)

sys.path.append(project_root)
