import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import datetime as dt
import pandas_datareader as web 

from sklearn.preprocessing import MinMaxScaler
import sklearn
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Dropout , LSTM
import tensorflow as tf

data = pd.read_csv('LAURUSLABS.csv')
print(data)

trainset = data.iloc[:,4:5]


length =  len(trainset)
traning_size = int(length*0.65)
test_size = length-traning_size

plt.plot(trainset)
plt.show()

print(traning_size)
print(test_size)

sc = MinMaxScaler(feature_range = (0,1))
training_scaled = sc.fit_transform(trainset.values.reshape(-1,1))
print(training_scaled)

train_data , test_data = training_scaled[0:traning_size,:],training_scaled[traning_size:length,:1]

import numpy
# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return numpy.array(dataX), numpy.array(dataY)

time_step = 15
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)

print(X_train.shape)
print(y_train.shape)

X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(X_train.shape[1],1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')


print(model.summary())
from sklearn.preprocessing import StandardScaler

model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,batch_size=10)



train_predict=model.predict(X_train)
test_predict=model.predict(X_test)

train_predict=sc.inverse_transform(train_predict)
test_predict=sc.inverse_transform(test_predict)

print(train_predict)
print(test_predict)

import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))

math.sqrt(mean_squared_error(ytest,test_predict))

look_back=15
trainPredictPlot = numpy.empty_like(training_scaled)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(training_scaled)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(training_scaled)-1, :] = test_predict
# plot baseline and predictions
plt.plot(sc.inverse_transform(training_scaled))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

print(len(test_data))   #349  #366


x_input = test_data[351:].reshape(1,-1)


temp_input=list(x_input)

temp_input=temp_input[0].tolist()


lst_output=[]
n_steps=15
i=0
while(i<8):
    
    if(len(temp_input)>15):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    

print(sc.inverse_transform(lst_output))

#print(sc.inverse_transform(lst_output))
print(trainset)
    