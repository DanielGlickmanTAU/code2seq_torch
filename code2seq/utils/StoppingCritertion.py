class StoppingCriterion:
    def __init__(self, threshold=None, desired_stable_evaluations=2, higher_is_better=True):
        self.threshold = threshold
        self.desired_stable_evaluations = desired_stable_evaluations
        self.higher_is_better = higher_is_better
        self.num_consecutive_successes = 0
        self.num_no_improvement = 0
        self.last_result = float('-inf') if higher_is_better else float('inf')

    def __call__(self, result):
        if self.threshold:
            if self.is_better(result, self.threshold):
                self.num_consecutive_successes += 1
            else:
                self.num_consecutive_successes = 0

        self.last_result = result
        return self.num_consecutive_successes >= self.desired_stable_evaluations

    def is_better(self, new_value, old_value):
        if self.higher_is_better:
            return new_value >= old_value
        return new_value <= old_value
