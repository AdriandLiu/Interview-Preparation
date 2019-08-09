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



# Part three prediction
# Get the real stock price
dataset_test = pd.read_csv("/home/donghan/DeepLearning/Recurrent_Neural_Networks/Google_Stock_Price_Test.csv")
read_stock_price = dataset_test.iloc[:,1:2].values


# Get the predicted stock price
dataset_total = pd.concat([dataset_train['Open'], dataset_test['Open']], axis=0)
inputs = dataset_total[len(dataset_total)-len(dataset_test)-60:].values
inputs = inputs.reshape(-1,1)
# (-1,1) Automatically calculate the number of vectors in row by column number
inputs = sc.transform(inputs)
# No fit_transform because sc already fitted with training data

X_test = []
for i in range(60,80):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1],1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


plt.plot(real_stock_price, color = "red", label = "Real Google Stock Price")
plt.plot(predicted_stock_price, color = "blue", label = "Predicted Google Stock Price")
plt.title("Google Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Google Stock Price")
plt.legend()
plt.show()
