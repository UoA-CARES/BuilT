
import pandas as pd     

from built.splitter import CsvSplitter
from built.registry import Registry

@Registry.register(category="splitter")
class TweetSplitter(CsvSplitter):
    def __init__(self, csv_path, n_splits=5, shuffle=True, random_state=42):
        self.csv_path = csv_path
        df = pd.read_csv(csv_path)
        super().__init__(df, df.sentiment, n_splits=n_splits, shuffle=shuffle, random_state=random_state)
