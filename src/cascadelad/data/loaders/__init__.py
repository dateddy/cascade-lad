from cascadelad.data.loaders import hdfs  # noqa: F401  (registers the loader)
from cascadelad.data.loaders.base import get_loader, register_loader

__all__ = ["get_loader", "register_loader"]
