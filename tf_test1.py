import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import time
import datetime

data = np.load('mass_spring_learning_data_norm_50k.npy').T
input = data[:, :-1]
output = data[:, -1]
train_num = 40000
x_tr = input[:train_num, :]
y_tr = output[:train_num]
x_te = input[train_num:, :]
y_te = output[train_num:]

nn = []
ls = 10
us = 10

nn.append(layers.Dense(us, activation='relu',input_shape=(x_tr.shape[1],)))
for i in range(ls):
    nn.append(layers.Dense(us, activation='relu'))
nn.append(layers.Dense(1))

model = keras.Sequential(nn)
model.summary()
optimizer = tf.keras.optimizers.RMSprop(0.001)
model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])

EPOCHS = 1500
time_0 = time.time()
t0 = datetime.time()
mses = []
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        time_now = time.time()
        time_pass = time_now - time_0
        time_left = time_pass*(EPOCHS-epoch-1)/(epoch+1)
        tp = datetime.timedelta(seconds=int(time_pass))
        tl = datetime.timedelta(seconds=int(time_left))
        mse = logs.get('mse')
        mses.append(mse)
        print('epoch:'+str(epoch),end='\t')
        print('MSE:'+str(round(mse, 8)), end='\t')
        print(str(tp)+' pass\t'+str(tl)+' left')

history = model.fit(x_tr, y_tr, 
                    epochs=EPOCHS, validation_split=0.2, verbose=0, 
                    callbacks=[PrintDot()])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()
print(hist)
plt.plot(mses)
plt.show()
# model.compile()