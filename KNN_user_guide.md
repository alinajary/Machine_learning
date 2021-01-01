KNN user guide:

# neccessary libraries:
1) Matplotlib
2) Numpy
3) Pandas
4) Seaborn
5) Scikit Learn

# Stpes
1) Import Libraries
2) Read in Dataset
3) Standardize scale to prep for KNN Algorithm
4) Split data into training and test sets
5) Create and Train the Model
6) Make predictions with the Model
7) Evaluate the predictions
8) Evaluate alternative K-values for better predictions
9) Plot Error Rate
10) Adjust K value per error rate evaluations  

Step 1: Import the necessary Libraries
Similar to previous discussions, we will need to import libraries that allow for data analysis and data visualization to get acclimated to the dataset. We will be using pandas, numpy, matplotlib and seaborn to conduct this.

Data Exploration libraries

`import pandas as pd`
`import numpy as np`

Data Visualization libraries
```
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```
# Step 2: Read in the dataset
We will use the pandas .read_csv() method to read in the dataset. Then we will use the .head() method to observe the first few rows of the data, to understand the information better. In our case, the feature(column) headers tell us pretty little. This is fine because we are merely trying to gain insight via classifying new data points by referencing it’s neighboring elements.
```
# Use pandas .read_csv() method to read in classified dataset
# index_col -> argument assigns the index to a particular 
columndf = pd.read_csv('Classified Data', index_col=0)
# Use the .head() method to display the first few rows
df.head()
```  


# Step 3: Standardize (normalize) the data scale to prep for KNN algorithm
Because the distance between pairs of points plays a critical part on the classification, it is necessary to normalize the data to minimize this(helpful link). This will generate an array of values. Again, KNN depends on the distance between each feature.

#Import module to standardize the scale
from sklearn.preprocessing import StandardScaler
#Create instance (i.e. object) of the standard scaler
scaler = StandardScaler()
#Fit the object to all the data except the Target Class
#use the .drop() method to gather all features except Target Class
#axis -> argument refers to columns; a 0 would represent rows
scaler.fit(df.drop('TARGET CLASS', axis=1))


# Use scaler object to conduct a transforms
scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))# Review the array of values generated from the scaled features process
scaled_features

# Step 4: Split the normalized data into training and test sets
This step is required to prepare us for the fitting (i.e. training) the model later. The “X” variable is a collection of all the features. The “y” variable is the target label which specifies the classification of 1 or 0 based. Our goal will be to identify which category the new data point should fall into.

#Import module to split the data
from sklearn.model_selection import train_test_split# Set the X and ys
X = df_feat
y = df['TARGET CLASS']# Use the train_test_split() method to split the data into respective sets
#test_size -> argument refers to the size of the test subset
#random_state -> argument ensures guarantee that the output of Run 
#1 will be equal to the output of Run 2, i.e. your split will be always the sameX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

This allows for use to train our model on the training set and evaluate the built model against the test set to identify errors.

# Step 5: Create and Train the Model
Here we create a KNN Object and use the .fit() method to train the model. Upon completion of the model we should receive confirmation that the training has been completed.

# Import module for KNN
from sklearn.neighbors import KNeighborsClassifier# Create KNN instance
# n_neighbors -> argument identifies the amount of neighbors used to ID classification
knn = KNeighborsClassifier(n_neighbors=1)# Fit (i.e. traing) the model
knn.fit(X_train, y_train)

# Step 6: Make Predictions
Here we review where our model was accurate and where it misclassified elements.

# Use the .predict() method to make predictions from the X_test subset
pred = knn.predict(X_test)# Review the predictions
pred

# Step 7: Evaluate the predictions
Evaluate the Model by reviewing the classification report or confusion matrix. By reviewing these tables, we are able to evaluate how accurate our model is with new values.

#Import classification report and confusion matrix to evaluate predictions
from sklearn.metrics import classification_report, confusion_matrix

Classification Report -> This tells us our model was 92% accurate…

#Print out classification report and confusion matrix
print(classification_report(y_test, pred))


# Print out confusion matrix
cmat = confusion_matrix(y_test, pred)#print(cmat)
print('TP - True Negative {}'.format(cmat[0,0]))
print('FP - False Positive {}'.format(cmat[0,1]))
print('FN - False Negative {}'.format(cmat[1,0]))
print('TP - True Positive {}'.format(cmat[1,1]))
print('Accuracy Rate: {}'.format(np.divide(np.sum([cmat[0,0],cmat[1,1]]),np.sum(cmat))))
print('Misclassification Rate: {}'.format(np.divide(np.sum([cmat[0,1],cmat[1,0]]),np.sum(cmat))))


# Step 8: Evaluate alternative K-values for better predictions
To simplify the process of evaluating multiple cases of k-values, we create a function to derive the error using the average where our predictions were not equal to the test values.

#Generate function to add error rates of KNN with various k-values
#error_rate -> empty list to gather error rates at various k-values
#for loop -> loops through k values 1 to 39
#knn -> creates instance of KNeighborsClassifier with various k
#knn.fit -> trains the model
#pred_i -> conducts predictions from model on test subset
#error_rate.append -> adds error rate of model with various k-value, using the average where prediction not
#equal to the test valueserror_rate = []for i in range(1,40):
    
knn = KNeighborsClassifier(n_neighbors=i)
knn.fit(X_train, y_train)
pred_i = knn.predict(X_test)
error_rate.append(np.mean(pred_i != y_test))

# Step 9: Plot Error Rate

# Configure and plot error rate over k valuesplt.figure(figsize=(10,4))
plt.plot(range(1,40), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K-Values')
plt.xlabel('K-Values')
plt.ylabel('Error Rate')

Here we see that the error rate continues to decrease as we increase the k-value. A picture tells a thousand words. Or at least here, we are able to understand what value of k leads to an optimal model. The k-value of 17 seems to give a decent error rate without too much noise, as we see with k-values of 28 and larger.

10) Adjust K value per error rate evaluations
This is just fine tuning our model to increase accuracy. We will need to retrain our model with the new k-value.
```
# Retrain model using optimal k-valueknn = KNeighborsClassifier(n_neighbors=17)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
```
Classification Report -> This tells us our model was 95% accurate…

# Print out classification report and confusion matrix
print(classification_report(y_test, pred))

```
#Print out confusion matrix
cmat = confusion_matrix(y_test, pred)#print(cmat)
print('TP - True Negative {}'.format(cmat[0,0]))
print('FP - False Positive {}'.format(cmat[0,1]))
print('FN - False Negative {}'.format(cmat[1,0]))
print('TP - True Positive {}'.format(cmat[1,1]))
print('Accuracy Rate: {}'.format(np.divide(np.sum([cmat[0,0],cmat[1,1]]),np.sum(cmat))))
print('Misclassification Rate: {}'.format(np.divide(np.sum([cmat[0,1],cmat[1,0]]),np.sum(cmat))))
```
Well, until next time. Still trying to find a better way to consolidate the information as I learn new content. The goal is to get back to publishing a post a day, per each new machine learning algorithm studied.


