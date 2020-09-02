import unittest
import yaml

yaml_config = """
program: train.py
method: grid
metric:
  goal: minimize
  name: loss
parameters:
  dropout:
    values: [0.1, 0.2, 0.4, 0.5, 0.7]
  channels_one:
    values: [10, 12, 14, 16, 18, 20]
  channels_two:
    values: [24, 28, 32, 36, 40, 44]
  learning_rate:
    values: [0.001, 0.005, 0.0005]
  epochs:
    value: 27
early_terminate:
  type: hyperband
  s: 2
  eta: 3
  max_iter: 27
"""

class SweepTest(unittest.TestCase):
    def test_build_sweep(self):
        hyperparam = yaml.load(yaml_config)
        print(hyperparam)
        

if __name__ == '__main__':
    unittest.main()
