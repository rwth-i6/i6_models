"""
i6_models root import
"""

from .__setup__ import get_version_str as _get_version_str

__long_version__ = _get_version_str(
    fallback="1.0.0+unknown", long=True, verbose_error=True
)  # `SemVer <https://semver.org/>`__ compatible
__version__ = __long_version__[: __long_version__.index("+")]  # distutils.version.StrictVersion compatible
__git_version__ = __long_version__  # just an alias, to keep similar to other projects
