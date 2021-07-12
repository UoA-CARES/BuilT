
from sklearn import model_selection
from built.utils.util_functions import seed_everything

class Splitter(object):
    def __init__(self, X, y, n_splits=5, shuffle=True, random_state=42):
        assert(X is not None)
        assert(y is not None)
        assert(len(X) == len(y))

        seed_everything(random_state)
                
        self.X = X
        self.y = y
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        # self.kf = model_selection.StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        self.kf = model_selection.KFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
    def get_fold(self, n_fold):
        assert(n_fold >= 0 and n_fold < self.n_splits)
        # for fold, (trn_, val_) in enumerate(self.kf.split(X=self.X, y=self.y)):
        for fold, (trn_, val_) in enumerate(self.kf.split(X=self.X)):
            if fold == n_fold:
                return trn_, val_

import pandas as pd     
import abc

class CsvSplitter(Splitter):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, X, y, n_splits=5, shuffle=True, random_state=42):
        # df = pd.read_csv(csv_path)
        super().__init__(X, y, n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    @abc.abstractmethod
    def get_X(self):
        pass

    @abc.abstractmethod
    def get_y(self):
        pass


    

