from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationConfig:
    """
    Immutable configuration for a model evaluation run.

    device: "auto" selects CUDA if available, else CPU.
            Pass "cpu" or "cuda" to override explicitly.
    top_k:  k for top-k accuracy (e.g. top-5 accuracy on CIFAR-10).
    """

    batch_size: int = 128
    device: str = "auto"
    top_k: int = 5
    num_workers: int = 2

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError(
                f"device must be 'auto', 'cpu', or 'cuda', got '{self.device}'"
            )