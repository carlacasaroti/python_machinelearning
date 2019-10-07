"""For an example, we're going to try and fit a non-linear model
to the datapoints corresponding to China's GDP from 1960 to 2014.
We download a dataset with two columns, the first, a year between 1960 and 2014,
the second, China's corresponding annual gross domestic income in US dollars for that year."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

 # reading te data
path = 'C:/Users/Usuario/Downloads/china_gdp.csv'
df = pd.read_csv(path)

# plotting dataset
plt.figure(figsize=(8,5))
x_data, y_data = (df["Year"].values, df["Value"].values)
plt.plot(x_data, y_data, 'ro')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()

# building the model
# let's build our regression model and initialize its parameters.
def sigmoid(x, Beta_1, Beta_2):
    y = 1 / (1 + np.exp(-Beta_1 * (x - Beta_2)))
    return y


# let's look at a sample sigmoid line that might fit with the data:
beta_1 = 0.10
beta_2 = 1990.0
# logistic function
Y_pred = sigmoid(x_data, beta_1 , beta_2)
# plot initial prediction against datapoints
plt.plot(x_data, Y_pred*15000000000000.)
plt.plot(x_data, y_data, 'ro')
plt.show()

# normalizing our data
xdata = x_data/max(x_data)
ydata = y_data/max(y_data)

# finding the best fit parameters using curve_fit which uses non-linear least squares
# to fit our sigmoid function, to data. Optimal values for the parameters so that
# the sum of the squared residuals of sigmoid(xdata, *popt) - ydata is minimized.
# popt are our optimized parameters
from scipy.optimize import curve_fit
popt, pcov = curve_fit(sigmoid, xdata, ydata)
#print the final parameters
print(" beta_1 = %f, beta_2 = %f" % (popt[0], popt[1]))

# plotting the regression model
x = np.linspace(1960, 2015, 55)
x = x/max(x)
plt.figure(figsize=(8,5))
y = sigmoid(x, *popt)
plt.plot(xdata, ydata, 'ro', label='data')
plt.plot(x,y, linewidth=3.0, label='fit')
plt.legend(loc='best')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()



