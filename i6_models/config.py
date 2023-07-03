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
from typing import Generic, TypeVar, Type
import inspect


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
class ModuleFactoryV1(Generic[ConfigType, ModuleType]):
    """
    Dataclass for a combination of a Subassembly/Part and the corresponding configuration.
    Also provides a function to construct the corresponding object through this dataclass
    """

    module_class: Type[ModuleType]
    cfg: ConfigType

    def __call__(self) -> ModuleType:
        """Constructs an instance of the given module class"""
        return self.module_class(self.cfg)

    def __post_init__(self) -> None:
        # Check typing of module_class and cfg, i.e. make sure that "self.module_class(self.cfg)" is a valid call.
        parameters = inspect.signature(self.module_class).parameters.values()
        assert len(parameters) >= 1
        parameter_iter = iter(parameters)

        # 1. Check that the first parameter is either not annotated or the annotation matches the type of self.cfg
        cfg_parameter = next(parameter_iter)
        if cfg_parameter.annotation is not inspect.Parameter.empty:
            typeguard.check_type(self.cfg, cfg_parameter.annotation)

        # 2. Check that all other parameters have default values
        for parameter in parameter_iter:
            assert parameter.default is not inspect.Parameter.empty
