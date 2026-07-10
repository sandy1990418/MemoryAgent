from dataclasses import dataclass


@dataclass(frozen=True)
class TokenBudget:
    max_tokens: int
    estimator_name: str = "chars_div_4"

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
