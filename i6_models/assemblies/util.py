from dataclasses import MISSING, dataclass
from typing import Any, List, Optional


@dataclass
class ModelConfiguration:
    """base dataclass that supported fetching attributes"""

    _name: Optional[str] = None

    @staticmethod
    def name() -> Optional[str]:
        return None

    def _get_all_attributes(self) -> List[str]:
        return [k for k in self.__dataclass_fields__.keys()]

    def _get_name(self, attribute_name: str) -> str:
        return self.__dataclass_fields__[attribute_name].name

    def _get_default(self, attribute_name: str) -> Any:
        if hasattr(self, attribute_name):
            return getattr(self, attribute_name)

        f = self.__dataclass_fields__[attribute_name]
        if f.default_factory is not MISSING:
            return f.default_factory()
        return f.default

    def _get_type(self, attribute_name: str) -> Any:
        return self.__dataclass_fields__[attribute_name].type

    @classmethod
    def from_namespace(cls, args):
        """
        Generates a ModelConfiguration from a given dataclass.
        If its already of the correct instance it will return the input object, otherwise it creates an object
        from the matching attributes.
        """
        if isinstance(args, cls):
            return args
        else:
            config = cls()
            for k in config.__dataclass_fields__.keys():
                if k.startswith("_"):
                    # private member, skip
                    continue
                if hasattr(args, k):
                    setattr(config, k, getattr(args, k))

            return config
