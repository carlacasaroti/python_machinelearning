import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
# %matplotlib inline

 # reading te data
path = 'C:/Users/Usuario/Downloads/FuelConsumption.csv'
df = pd.read_csv(path)

# selecting some of the available features to build the model
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

# creating training and test sets
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

"""PolynomialFeatures() function in Scikit-learn library, drives a new feature sets from the original feature set. 
   That is, a matrix will be generated consisting of all polynomial combinations of the features with degree less than 
   or equal to the specified degree. For example, lets say the original feature set has only one feature, ENGINESIZE. 
   Now, if we select the degree of the polynomial to be 2, then it generates 3 features, 
   degree=0, degree=1 and degree=2:"""

# setting and training the model
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])

poly = PolynomialFeatures(degree=2)
train_x_poly = poly.fit_transform(train_x)

# now using what was calculated by the polynomial equation before
# it is a matter of using the data calculated in train_x_poly into a multiple linear regression
clf = linear_model.LinearRegression()
train_y_ = clf.fit(train_x_poly, train_y)
# The coefficients
print ('Coefficients: ', clf.coef_)
print ('Intercept: ',clf.intercept_)

"""Coefficient and Intercept , are the parameters of the fit curvy line. 
Given that it is a typical multiple linear regression, with 3 parameters, 
and knowing that the parameters are the intercept and coefficients of hyperplane,
sklearn has estimated them from our new set of feature sets. Lets plot it:"""
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
XX = np.arange(0.0, 10.0, 0.1)
yy = clf.intercept_[0]+ clf.coef_[0][1]*XX+ clf.coef_[0][2]*np.power(XX, 2)
plt.plot(XX, yy, '-r' )
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()
# evaluation
from sklearn.metrics import r2_score

test_x_poly = poly.fit_transform(test_x)
test_y_ = clf.predict(test_x_poly)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_ , test_y) )