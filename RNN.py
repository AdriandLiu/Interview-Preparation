import numpy as np
import matplotlib.pylab as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Import training set
dataset_train = pd.read_csv("/home/donghan/DeepLearning/Recurrent_Neural_Networks/Google_Stock_Price_Train.csv")
training_set = dataset_train.iloc[:,1:2].values

# Feature scaling
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)


# Creating a data structure with 60 timesteps and 1 output
# 60 timesteps of past information to learn to train in the next step

X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

# Reshape
X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
# data, shape, number of predictors



# Initializing Recurrent_Neural_Networks
regressor = Sequential()

# First LSTM layer
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Second LSTM layers
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Third LSTM layers
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Fourth LSTM lsyers
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Ouput layer
regressor.add(Dense(units = 1))

# Compile
regressor.compile(optimizer = "adam", loss = "mean_squared_error")
# OR RMSprop

# Fit LSTM
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
