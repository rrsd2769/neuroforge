from abc import ABC, abstractmethod
from typing import Any

from neuroforge_core.domain.entities.architecture import Architecture


class IModelCompiler(ABC):
    """
    Port: translate a domain Architecture into a backend-specific model.

    The return type is Any because the domain layer must not import
    torch. The infrastructure adapter (PyTorchModelCompiler) narrows
    this to torch.nn.Module.
    """

    @abstractmethod
    def compile(
        self,
        architecture: Architecture,
        input_channels: int = 3,
        num_classes: int = 10,
    ) -> Any:
        """
        Compile an Architecture into a trainable model object.

        Raises:
            ModelCompilationError: if the architecture is structurally invalid.
        """
        ...

    @abstractmethod
    def is_valid_architecture(
        self,
        architecture: Architecture,
        input_channels: int = 3,
        num_classes: int = 10,
    ) -> bool:
        """
        Return True if the architecture compiles and produces valid output shape.
        Does not raise — designed for use in search loops where invalid
        architectures are common and expected.
        """
        ...