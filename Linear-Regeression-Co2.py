#Importing Needed packages
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

# Downloading Data using colab
from google.colab import files
import io
uploaded = files.upload()

#Readind the data
df= pd.read_csv(io.BytesIO(uploaded['FuelConsumptionCo2.csv']))
df.head()

#Data Exploration 
df.describe() # summarize the data

#Selecte just 4 features to explore more
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)

#Plot features in histogramme
viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
viz.hist()
plt.show()

#Plot each of those features against the CO2EMISSIONS

#1-FUELCONSUMPTION_COMB vs CO2EMISSIONS
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()

#2-ENGINESIZE vs CO2EMISSIONS
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

#3-CYLINDERS vs CO2EMISSIONS 
plt.scatter(cdf.CYLINDERS , cdf.CO2EMISSIONS, color='blue')
plt.xlabel("cylinder")
plt.ylabel("Emissions")
plt.show()


# Creating train and test dataset
#Let's split our dataset into train and test sets. 80% of the entire dataset will be used for training and 20% for testing
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk] #training DataSet
test = cdf[~msk] #testing DataSet
train.head(3)
test.head(3)

#I-Model to predict CO2EMISSIONS from ENGINESIZE

#Modeling Using sklearn package to model data
from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)
# Coefficient and Intercept in the simple linear regression, are the parameters of the fit line
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)

#Plot outputs
#Train data distribution
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
#Fit line
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
#Variable x et y
plt.xlabel("Engine size")
plt.ylabel("Emission") #target

#Evaluation the Model
from sklearn.metrics import r2_score
#testing DataSet
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
#Prediction
test_y_ = regr.predict(test_x)
#Evaluation
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y , test_y_) )


#II-Model to predict CO2EMISSIONS from FUELCONSUMPTION_COMB

#split Data
train_x= train[["FUELCONSUMPTION_COMB"]]
test_x =test[["FUELCONSUMPTION_COMB"]]

#Model
regr =linear_model.linearRegression() # in sklearn
regr.fit (train_x ,train_y) # train_y it's training Dataset on target CO2EMISSINS

#Predictions
predictions =regr.predict(test_x)

#Evaluation 
print("R2-score: %.2f" % r2_score(test_y , predictions) )



