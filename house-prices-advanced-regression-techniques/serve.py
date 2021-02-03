import pandas as pd
import numpy as np
import tensorflow as tf
from data import *
from sklearn.preprocessing import StandardScaler

data_df = csv_data(filename='test').cat_df()
mals_df = pd.read_csv(modulepath + 'mal_features.csv', header=None).to_dict()
mals = list(mals_df[1].values())
features = data_df.drop(columns=mals)

X = features.to_numpy()
X = StandardScaler().fit(X).transform(X)

model = tf.keras.models.load_model(modulepath + 'model.h5')
predictions = model.predict(x=X)
saleprices = []
for prediction in predictions:
    saleprices.append(prediction[0])

submission = pd.DataFrame(data={'SalePrice': saleprices}, index=data_df.index)
submission = submission.replace(r'\s+', np.nan, regex=True)
submission = submission.fillna(0)
submission.to_csv(modulepath+'submission.csv')

