"""
Provides the base class for configurations, model configurations can be derived from this base class e.g:

@dataclass
class ExampleConfig(ModelConfiguration):
    hidden_dim: int = 256
    name: str = "Example Configuration"

class ExampleModule(Module):
    __init__(self, cfg: ExampleConfig)
        self.hidden_dim = cfg.hidden_dim

This config can then be used in the construction of the model to provide parameters.
Approach is inspired by the Fairseq: https://github.com/facebookresearch/fairseq/blob/main/fairseq/dataclass/configs.py
"""

from dataclasses import MISSING, dataclass
from typing import Any, List


@dataclass
class ModelConfiguration:
    """
    Base dataclass that supports fetching attributes
    """

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
        If it's already of the correct instance, it will return the input object, otherwise it creates an object
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
