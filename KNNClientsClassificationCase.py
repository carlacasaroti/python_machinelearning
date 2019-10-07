""" K-Nearest Neighbors is an algorithm for supervised learning.
Where the data is 'trained' with data points corresponding to their classification.
Once a point is to be predicted, it takes into account the 'K' nearest points to it to determine it's classification.
"""

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing



"""About the dataset Imagine a telecommunications provider has segmented its customer base by service usage patterns, 
categorizing the customers into four groups. If demographic data can be used to predict group membership, 
the company can customize offers for individual prospective customers. It is a classification problem. 
That is, given the dataset, with predefined labels, we need to build a model to be used 
to predict class of a new or unknown case. The example focuses on using demographic data, such as region, 
age, and marital, to predict usage patterns. The target field, called custcat, has four possible values 
that correspond to the four customer groups, as follows: 1- Basic Service 2- E-Service 3- Plus Service 4- Total Service
Our objective is to build a classifier, to predict the class of unknown cases.
We will use a specific type of classification called K nearest neighbour."""

# uploading the data frame, which is the clients, their features and their classifications
path = "C:/Users/Usuario/Documents/_2019/Machine Learning Coursera/teleCust1000t.csv"
df = pd.read_csv(path)
#print df.head()

# splitting the dataset into features and given classes
list_columns = df.columns
# taking all the feature columns - exception of the last on containing y
X = df[list_columns[0:-1]].values
# now separating the classes
y = df[list_columns[-1]].values

# as good practice, let's standardize the features
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

# next step is to separate the test and train sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
# print ('Train set:', X_train.shape,  y_train.shape)
# print ('Test set:', X_test.shape,  y_test.shape)

# in order to choose the best K value, let's have a look on the following graph:
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
Ks = 10
mean_acc = np.zeros((Ks - 1))
std_acc = np.zeros((Ks - 1))
ConfustionMx = []

for n in range(1, Ks):
    # Train Model and Predict
    neigh = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)
    yhat = neigh.predict(X_test)
    mean_acc[n - 1] = metrics.accuracy_score(y_test, yhat)

    std_acc[n - 1] = np.std(yhat == y_test) / np.sqrt(yhat.shape[0])

#print mean_acc

# now lets have a look to it graphically
plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks), mean_acc - 1 * std_acc, mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()

print "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1

# as we can visually and numerically remark, the best accuracy value is given using k=9
# in that way, to train and test the model we will use k=9
k = 9
# train model and predict using the optimal k
neigh9 = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
yhat9 = neigh9.predict(X_test)

# evaluating the model accuracy
print "Train set Accuracy: ", metrics.accuracy_score(y_train, neigh9.predict(X_train))
print "Test set Accuracy: ", metrics.accuracy_score(y_test, yhat9)

