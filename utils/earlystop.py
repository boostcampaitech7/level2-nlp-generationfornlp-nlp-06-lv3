from transformers import TrainerCallback


class CustomEarlyStoppingCallback(TrainerCallback):
    def __init__(self, monitor, patience, threshold, greater_is_better):
        self.monitor = monitor
        self.patience = patience
        self.threshold = threshold
        self.greater_is_better = greater_is_better
        self.best_score = None
        self.num_bad_epochs = 0

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        current_score = metrics.get(self.monitor)
        if current_score is None:
            return

        if self.best_score is None:
            self.best_score = current_score
            return

        if self.greater_is_better:
            improvement = current_score - self.best_score
        else:
            improvement = self.best_score - current_score

        if improvement > self.threshold:
            self.best_score = current_score
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs >= self.patience:
                control.should_training_stop = True
