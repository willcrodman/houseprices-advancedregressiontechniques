import tensorflow as tf
from data import *
from features import mal_features
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data = csv_data(filename='train')
data_df = data.cat_df()
labels = data_df.filter(['SalePrice'])
if path.exists(modulepath + 'mal_features.csv') == False:
    mals_cls = mal_features(max_na_threshold=0.15, max_corr_threshold=0.75)
    mals = mals_cls(data=data)
else:
    mals_df = pd.read_csv(modulepath + 'mal_features.csv', header=None).to_dict()
    mals = list(mals_df[1].values())
mals.append('SalePrice')
features = data_df.drop(columns=mals)

X_train, X_test, y_train, y_test =\
    train_test_split(features, np.ravel(labels), test_size=0.1, random_state=42)
scaler = MinMaxScaler(feature_range=(-1, 1))
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
print(X_train[10])
pd.Series(data=X_train[10]).to_csv(path_or_buf=modulepath + 'features.csv')
k = tf.keras
input_shape = len(features.keys())
def model():
    model = tf.keras.Sequential([
        k.layers.Dense(input_shape * 20, activation='relu', input_shape=[input_shape]),
        k.layers.Dense(input_shape * 6, activation='relu'),
        k.layers.Dense(input_shape * 3, activation='relu'),
        k.layers.Dense(1),
        k.layers.Dropout(0.2)
    ])
    optimizer = k.optimizers.Adam(learning_rate=0.0001, clipvalue=0.5)
    model.compile(optimizer=optimizer, loss='mae', metrics=['mape', 'mae'])
    return model
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_mape', min_delta=0, patience=100, mode='auto',
        baseline=None, restore_best_weights=False)
]
if path.exists(modulepath + 'model.h5') == False:
    mdl = model()
    history = mdl.fit(X_train, y_train, epochs=1000, batch_size=10,
                      validation_split=0.2, callbacks=callbacks)
    #mdl.save(filepath=modulepath + 'model.h5')
    '''
    for metric in history.history.keys():
        plt.plot(history.history[metric])
        plt.title(f'model {metric}')
        plt.ylabel(metric)
        plt.xlabel('epoch')
        plt.show()
    '''
else:
    mdl = tf.keras.models.load_model(filepath=modulepath + 'model.h5')

results = mdl.evaluate(X_test, y_test, batch_size=128)
loss, mean_absolute_percentage_error, mean_absolute_error =\
    results[0], results[1], results[2]
print(mean_absolute_percentage_error)