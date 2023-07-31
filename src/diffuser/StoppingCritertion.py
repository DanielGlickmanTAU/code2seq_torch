class StoppingCriterion:
    def __init__(self, threshold=None, desired_stable_evaluations=2, patience=None, higher_is_better=True):
        self.threshold = threshold
        self.desired_stable_evaluations = desired_stable_evaluations
        self.patience = patience
        self.higher_is_better = higher_is_better
        self.num_consecutive_successes = 0
        self.num_no_improvement = 0
        self.best_so_far = float('-inf') if higher_is_better else float('inf')

    def __call__(self, result):
        if self.threshold:
            if self.is_better(result, self.threshold):
                self.num_consecutive_successes += 1
            else:
                self.num_consecutive_successes = 0

        if self.is_better(self.best_so_far, result):
            self.num_no_improvement += 1
        else:
            self.num_no_improvement = 0
            self.best_so_far = result

        patience_critirion = self.patience and self.num_no_improvement > self.patience
        thershold_critiertion = self.desired_stable_evaluations and self.num_consecutive_successes >= self.desired_stable_evaluations
        return thershold_critiertion or patience_critirion

    def is_better(self, new_value, old_value):
        if self.higher_is_better:
            return new_value >= old_value
        return new_value <= old_value
