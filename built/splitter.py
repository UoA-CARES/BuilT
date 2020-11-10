
from sklearn import model_selection

class Splitter(object):
    def __init__(self, X, y, n_splits=5, shuffle=True, random_state=42):
        assert(X is not None)
        assert(y is not None)
        assert(len(X) == len(y))

        self.X = X
        self.y = y
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.kf = model_selection.StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)

    def get_fold(self, n_fold):
        assert(n_fold >= 0 and n_fold < self.n_splits)
        for fold, (trn_, val_) in enumerate(kf.split(X=train, y=train.sentiment.values)):
            if fold == n_fold:
                return trn_, val_
    
