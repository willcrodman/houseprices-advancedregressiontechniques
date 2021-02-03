import pandas as pd
from os import path
import numpy as np

modulepath = '/Users/willrodman/Desktop/house-prices-advanced-regression-techniques/'

class csv_data():
    def __init__(self, filename):
        global modulepath
        self.filepath = modulepath + filename + '.csv'
        if path.exists(self.filepath):
            self.data = pd.read_csv(self.filepath, index_col='Id')

    def df(self):
        return self.data

    def cat_df(self):
        data_cat = pd.DataFrame()
        for feature, dtype in self.data.dtypes.items():
            if dtype == np.dtype(object):
                data_cat[feature] = self.data[feature].astype('category').cat.codes
            else:
                data_cat[feature] = self.data[feature]
        return data_cat





