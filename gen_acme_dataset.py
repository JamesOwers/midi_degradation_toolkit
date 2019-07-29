

# Dev mode warnings
import sys

if not sys.warnoptions:
    import os, warnings
    warnings.simplefilter("always") # Change the filter in this process
    os.environ["PYTHONWARNINGS"] = "always" # Also affect subprocesses