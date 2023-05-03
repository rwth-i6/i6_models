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
Similar approach as done in Fairseq: https://github.com/facebookresearch/fairseq/blob/main/fairseq/dataclass/configs.py
"""

from __future__ import annotations
from typing import Any
from dataclasses import dataclass, fields
import typeguard


@dataclass
class ModelConfiguration:
    """
    Base dataclass that supports fetching attributes.
    In order to enforce typing in a Config, decorate that Config with @enforce_types.
    """

    @classmethod
    def from_namespace(cls, args: Any) -> ModelConfiguration:
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

    def _validate_types(self) -> None:
        for field in fields(type(self)):
            try:
                typeguard.check_type(getattr(self, field.name), field.type)
            except typeguard.TypeCheckError as exc:
                raise typeguard.TypeCheckError(f'In field "{field.name}" of "{type(self)}": {exc}')

    def __post_init__(self) -> None:
        self._validate_types()
