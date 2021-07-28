
class EvaluationScheduler(object):
    def __init__(self, boundary_scores, intervals):
        self.boundary_scores = boundary_scores
        self.intervals = intervals
        self.last_eval_step = 0
        self.current_eval_interval = self.intervals[0]

    def scheduled(self, step):
        return step >= self.last_eval_step + self.current_eval_interval

    def update(self, step, score):
        for bs, interval in zip(self.boundary_scores, self.intervals):
            if score >= bs:
                # if interval < self.current_eval_interval:
                self.current_eval_interval = interval
                self.last_eval_step = step
                print(f'interval={self.current_eval_interval}, last_eval_step={self.last_eval_step}')
                break



        

