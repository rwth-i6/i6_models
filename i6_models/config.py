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
from dataclasses import dataclass, fields
import typeguard
from torch import nn
from typing import Callable, Generic, TypeVar


@dataclass
class ModelConfiguration:
    """
    Base dataclass for configuration of different primitives, parts and assemblies.
    Enforces type checking for the creation of derived configs.
    """

    def _validate_types(self) -> None:
        for field in fields(type(self)):
            try:
                if isinstance(field.type, str):
                    # Later Python versions use forward references,
                    # and then field.type is a str object.
                    # This needs to be resolved to the type itself.
                    # At the moment is does not look like there is typeguard support for it, thus manual fixes.
                    # C.f. https://github.com/agronholm/typeguard/issues/358
                    from typeguard._functions import check_type_internal
                    from typeguard._memo import TypeCheckMemo
                    from typing import ForwardRef
                    import sys

                    cls_globals = vars(sys.modules[self.__class__.__module__])
                    memo = TypeCheckMemo(globals=cls_globals, locals=cls_globals)
                    t = ForwardRef(field.type)
                    check_type_internal(getattr(self, field.name), t, memo)
                else:
                    typeguard.check_type(getattr(self, field.name), field.type)
            except typeguard.TypeCheckError as exc:
                raise typeguard.TypeCheckError(f'In field "{field.name}" of "{type(self)}": {exc}')

    def __post_init__(self) -> None:
        self._validate_types()


ConfigType = TypeVar("ConfigType", bound=ModelConfiguration)
ModuleType = TypeVar("ModuleType", bound=nn.Module)


@dataclass
class SubassemblyFactory(Generic[ConfigType, ModuleType]):
    """
    Dataclass for a combination of a Subassembly/Part and the corresponding configuration.
    Also provides a function to construct the corresponding object through this dataclass
    """

    module_class: Callable[[ConfigType], ModuleType]
    cfg: ConfigType

    def __call__(self) -> ModuleType:
        """Constructs an instance of the given module class"""
        return self.module_class(self.cfg)
