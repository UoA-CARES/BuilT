
class EvaluationScheduler(object):
    def __init__(self, boundary_scores, intervals):
        self.boundary_scores = boundary_scores
        self.intervals = intervals
        self.last_eval_step = -1
        self.__total_step = None
        self.current_eval_interval = None

    def scheduled(self, step):
        assert(self.current_eval_interval is not None)
        
        # step starts from 0, so we need to add 1
        return step >= self.last_eval_step + self.current_eval_interval

    def update(self, step, score):
        assert(self.__total_step is not None)
        self.last_eval_step = step
        
        for bs, interval in zip(self.boundary_scores, self.intervals):
            if score >= bs:
                if interval == 0:
                    self.current_eval_interval = self.total_step
                elif interval < self.current_eval_interval:
                    self.current_eval_interval = interval
                else:
                    # do nothing, keep the current interval
                    pass
                
        # print(f'interval={self.current_eval_interval}, last_eval_step={self.last_eval_step}')

    @property
    def total_step(self) -> int:
        return self.__total_step

    @total_step.setter
    def total_step(self, total_step: int):
        self.__total_step = total_step
        
        if self.current_eval_interval is None and self.intervals[0] == 0:
            self.current_eval_interval = self.__total_step
        elif self.current_eval_interval is None:
            self.current_eval_interval = self.intervals[0]
        else:
            pass

        

        

