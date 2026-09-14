"""Early stopping on a monitored metric that should decrease (e.g. val loss)."""


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.num_bad_epochs = 0

    def step(self, score):
        if self.best_score is None or score < self.best_score - self.min_delta:
            self.best_score = score
            self.num_bad_epochs = 0
            return False

        self.num_bad_epochs += 1
        return self.num_bad_epochs >= self.patience
