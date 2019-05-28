import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

# get the dataset using the link or download the dataset using curl in your Terminal
curl -O https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv

df = pd.read_csv('FuelConsumptionCo2.csv')
df.head()

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)

#to check whether linear regression good for making predictions on this data

plt.scatter(cdf.ENGINESIZE,cdf.CO2EMISSIONS,color='blue')
plt.xlabel('Engine Size')
plt.ylabel('Emission')
plt.show()

# Create a Training and Testing Dataset¶
# I'll be using 80% of the data for Training and the rest for testing. It will be mutually exclusive
msk = np.random.rand(len(df))<0.8
train = cdf[msk]
test =cdf[~msk]

# Train Data Distribution

plt.scatter(train.ENGINESIZE,train.CO2EMISSIONS, color='blue')
plt.xlabel('EngineSize')
plt.ylabel('C02Emissions')
plt.show()


#Multiple Regression Model
#We will be using Fuel Consumption, Cylinders and Engine size to predict the CO2 Emissions.

from sklearn import linear_model as lm
regr= lm.LinearRegression()
x= np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y=np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(x,y)
#THE COEFFICIENTS
print('Coefficients :', regr.coef_)


#Prediction

from sklearn.metrics import r2_score
y_hat = regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
x_test= np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y_test=np.asanyarray(test[['CO2EMISSIONS']])
#rse
print('RSE : %.2f' % np.mean((y_hat - y_test)**2))
#variance score : 1 is perfect
print('Variance Score: %.2f' %regr.score(x_test,y_test))
#R^2 Score
print("R2-score: %.2f" % r2_score(y_hat , y_test) )


#With independent variables as Fuel Consumption on City and Highway instead of combined.

from sklearn import linear_model as lm
regr= lm.LinearRegression()
x= np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_HWY','FUELCONSUMPTION_CITY']])
y=np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(x,y)
#THE COEFFICIENTS
print('Coefficients :', regr.coef_)

y_hat = regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_HWY','FUELCONSUMPTION_CITY']])
x_test= np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_HWY','FUELCONSUMPTION_CITY']])
y_test=np.asanyarray(test[['CO2EMISSIONS']])
#rse
print('RSE : %.2f' % np.mean((y_hat - y_test)**2))
#variance score : 1 is perfect
print('Variance Score: %.2f' %regr.score(x_test,y_test))
