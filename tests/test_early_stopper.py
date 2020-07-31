import unittest
from src.early_stopper import EarlyStopper


class TestEarlyStopper(unittest.TestCase):
    def test_max(self):
        es = EarlyStopper(patience=3, mode="max", delta=0.1)

        # first score
        stop, save = es(10)
        self.assertEqual(stop, False)
        self.assertEqual(save, True)

        # got improvement
        stop, save = es(20)
        self.assertEqual(stop, False)
        self.assertEqual(save, True)
        
        # got improvement
        stop, save = es(20.1)
        self.assertEqual(stop, False)
        self.assertEqual(save, True)

        # patience 1, still going
        stop, save = es(20.09)
        self.assertEqual(stop, False)
        self.assertEqual(save, False)

        # patience 2, still going
        stop, save = es(20.05)
        self.assertEqual(stop, False)
        self.assertEqual(save, False)

        # out of patience, stop it
        stop, save = es(20.01)
        self.assertEqual(stop, True)
        self.assertEqual(save, False)
        
    def test_min(self):
        es = EarlyStopper(patience=3, mode="min", delta=0.1)

        # first score
        stop, save = es(10)
        self.assertEqual(stop, False)
        self.assertEqual(save, True)

        # got improvement
        stop, save = es(9)
        self.assertEqual(stop, False)
        self.assertEqual(save, True)

        # got improvement
        stop, save = es(8.9)
        self.assertEqual(stop, False)
        self.assertEqual(save, True)

        # patience 1, still going
        stop, save = es(8.91)
        self.assertEqual(stop, False)
        self.assertEqual(save, False)

        # patience 2, still going
        stop, save = es(8.95)
        self.assertEqual(stop, False)
        self.assertEqual(save, False)

        # out of patience, stop it
        stop, save = es(8.91)
        self.assertEqual(stop, True)
        self.assertEqual(save, False)


if __name__ == '__main__':
    unittest.main()
