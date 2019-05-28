import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

curl -O https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv

df = pd.read_csv("FuelConsumptionCo2.csv")

df.head()

df.describe()

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)

viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
viz.hist()
plt.show()

plt.scatter(cdf.FUELCONSUMPTION_COMB,cdf.CO2EMISSIONS, color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel('Emission')
plt.show()

plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

#scatter plot of Cylinder Size v CO2 Emissions, the one we'll use.
plt.scatter(cdf.CYLINDERS,cdf.CO2EMISSIONS,color='blue',alpha=0.3)
plt.xlabel('Cylinder')
plt.ylabel('CO2 EMISSION')
plt.show()

#creating test and train dataset using numpy

msk = np.random.rand(len(cdf))<0.8
train= cdf[msk]
test= cdf[~msk]

# train data distribution
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='red')
plt.xlabel("Engine Size")
plt.ylabel("Emission")
plt.show()

#modeling
from sklearn import linear_model as lm
regr= lm.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)

#the coefficients
print("Coefficients :", regr.coef_)
print('Intercept :', regr.intercept_)


plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='red')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0],'-b')
plt.xlabel("Engine size")
plt.ylabel("Emission")

# R^2 Score, Higher it is, better the fit of the model is.
from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_hat = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_hat , test_y) )
