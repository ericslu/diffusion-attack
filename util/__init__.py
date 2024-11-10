import importlib
import glob
from os.path import dirname, basename, isfile, join

# Gather all .py files in the util directory except __init__.py
modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]

# Dynamically import each module in __all__
for module_name in __all__:
    try:
        # Use importlib to import the module with the correct package
        importlib.import_module(f"util.{module_name}")
    except ImportError as e:
        print(f"Could not import module {module_name}: {e}")
