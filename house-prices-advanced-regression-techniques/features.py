import pandas as pd
from data import *

class mal_features():
    mals = []

    def __init__(self, max_na_threshold=1, max_corr_threshold=1,
                 corr_coefficient_method='pearson'):
        self.max_na_threshold = max_na_threshold
        self.max_corr_threshold = max_corr_threshold
        self.corr_coefficient_method = corr_coefficient_method

    def __call__(self, *args, **kwargs):
        data = kwargs['data'].df().drop(columns=['SalePrice'])
        data_cat = kwargs['data'].cat_df().drop(columns=['SalePrice'])
        for feature, na in data.isna().mean().items():
            if na >= self.max_na_threshold:
                self.append(feature)
        for y_idx, row in data_cat.corr(method=self.corr_coefficient_method).iteritems():
            for x_idx, value in row.items():
                if value >= self.max_corr_threshold and value != 1.0:
                    self.append(y_idx, x_idx)
        if 'print' in args:
            print(f'For 79 features total. {(len(self.mals))} have been removed.'
                f'Features removed: {self.mals}')
        pd.Series(data=self.mals).to_csv(path_or_buf=modulepath + 'mal_features.csv')
        return self.mals

    @classmethod
    def append(cls, *features):
        for feature in features:
            if feature not in cls.mals:
                cls.mals.append(feature)