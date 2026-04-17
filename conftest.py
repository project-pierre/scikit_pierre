import sys
import os

# Ensure the project root is in sys.path so `import scikit_pierre` resolves
# to the inner library package (scikit_pierre/scikit_pierre/).
_root = os.path.dirname(os.path.abspath(__file__))
if _root not in sys.path:
    sys.path.insert(0, _root)
