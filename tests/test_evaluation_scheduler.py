import unittest
from built.evaluation_scheduler import EvaluationScheduler


class TestEvaluationScheduler(unittest.TestCase):
    def test_eval_on_each_epoch(self):
        eval_scores = [0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 0.99]

        # 0 <= score < 0.5, then interval is 0(it means the total_batch_step)
        # 0.5 <= score < 0.6, then interval is 0(it means the total_batch_step)
        # 0.6 <= score < 0.7, then interval is 0(it means the total_batch_step)
        # 0.7 <= score < 0.8, then interval is 0(it means the total_batch_step)
        # 0.8 <= score < 0.9, then interval is 0(it means the total_batch_step)
        # 0.9 <= score, then interval is 1
        boundary_scores = [0, 0.5, 0.6, 0.7, 0.8, 0.9]
        intervals = [0, 0, 0, 0, 0, 0]
        total_batch_step = 10
        num_of_epoch = 6

        es = EvaluationScheduler(boundary_scores, intervals)

        for i in range(num_of_epoch):
            epoch = i

            # first score
            es.total_step = total_batch_step

            for j in range(total_batch_step):
                step = j
                schedule_counter = (total_batch_step * epoch) + step
                if es.scheduled(schedule_counter):
                    eval_score = eval_scores[i]
                    print(
                        f'evaluation step:{step} / epoch:{epoch} / score:{eval_score}')
                    self.assertEqual(
                        step, total_batch_step - 1, "if the interval is set to 0, evaluation should be done at the last batch step.")
                    es.update(schedule_counter, eval_score)
            print()

    def test_eval_on_each_minibatch(self):
        eval_scores = [0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 0.99]

        # 0 <= score < 0.5, then interval is 1
        # 0.5 <= score < 0.6, then interval is 1
        # 0.6 <= score < 0.7, then interval is 1
        # 0.7 <= score < 0.8, then interval is 1
        # 0.8 <= score < 0.9, then interval is 1
        # 0.9 <= score, then interval is 1
        boundary_scores = [0, 0.5, 0.6, 0.7, 0.8, 0.9]
        intervals = [1, 1, 1, 1, 1, 1]
        total_batch_step = 10
        num_of_epoch = 6

        es = EvaluationScheduler(boundary_scores, intervals)

        for i in range(num_of_epoch):
            epoch = i

            # first score
            es.total_step = total_batch_step

            eval_cnt = 0
            for j in range(total_batch_step):
                step = j
                schedule_counter = (total_batch_step * epoch) + step

                if es.scheduled(schedule_counter):
                    eval_score = eval_scores[i]
                    print(
                        f'evaluation step:{step} / epoch:{epoch} / score:{eval_score}')

                    es.update(schedule_counter, eval_score)
                    eval_cnt += 1

            self.assertEqual(eval_cnt, total_batch_step,
                             "if the interval is set to 1, evaluation should be run at each batch step.")
            print()


if __name__ == '__main__':
    unittest.main()
