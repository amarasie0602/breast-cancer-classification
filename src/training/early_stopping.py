"""Early stopping on a monitored metric that should decrease (e.g. val loss)."""

from typing import Optional


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score: Optional[float] = None
        self.num_bad_epochs = 0

    def step(self, score: float) -> bool:
        if self.best_score is None or score < self.best_score - self.min_delta:
            self.best_score = score
            self.num_bad_epochs = 0
            return False

        self.num_bad_epochs += 1
        return self.num_bad_epochs >= self.patience
