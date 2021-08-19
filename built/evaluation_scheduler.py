
class EvaluationScheduler(object):
    def __init__(self, boundary_scores, intervals):
        self.boundary_scores = boundary_scores
        self.intervals = intervals
        self.last_eval_step = 0
        self.__total_step = None
        self.current_eval_interval = None

    def scheduled(self, step):
        assert(self.current_eval_interval is not None)
        
        # step starts from 0, so we need to add 1
        return step + 1 >= self.last_eval_step + self.current_eval_interval

    def update(self, step, score):
        assert(self.__total_step is not None)
        for bs, interval in zip(self.boundary_scores, self.intervals):
            if score >= bs:
                # if interval < self.current_eval_interval:
                if interval == 0:
                    self.current_eval_interval = self.total_step
                else:
                    self.current_eval_interval = interval
                self.last_eval_step = step
                # print(f'interval={self.current_eval_interval}, last_eval_step={self.last_eval_step}')
                break

    @property
    def total_step(self) -> int:
        return self.__total_step

    @total_step.setter
    def total_step(self, total_step: int):
        self.__total_step = total_step
        if self.intervals[0] == 0:
            self.current_eval_interval = self.__total_step
        else:
            self.current_eval_interval = self.intervals[0]

        

