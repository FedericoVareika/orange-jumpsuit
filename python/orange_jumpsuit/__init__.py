# Expose the IndexPQ class at the top level of the package
from .wrapper import IndexPQ

# Optional: define what gets imported if someone uses `from jumpsuit import *`
__all__ = ["IndexPQ"]
