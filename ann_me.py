import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#dataset in same file directory
dtst = pd.read_csv("Churn_Modelling.csv")

X = dtst.iloc[:, 3:13].values
Y = dtst.iloc[:, 13].values


# encoding 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]
#spliting dataset
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size =0.2,random_state=0)
#feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test)

# encoding or prepration of data encoding

# making of actual nural nets

#import library
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
# there are 2 ways either defining the sequential of layers or defining graph
# here it is by defining sequential of layers


#initializing ann
classifier = Sequential()

""" atually there r 7 steps in ann 1st is 
    randomly initialize weights to small number close to 0 but not 0
    which can be done with dense module
    step 2 input to the nural nets ie 11 idependet variable which we encoded before
    step 3 propogation
    step 4 find error
    step 5 back propogation
    step 6 repeat 1-5
    step 7 epoc (research needed)
"""
#Adding the input layer and first hidden layer
#dense function important units  average of number of input node and no of out put node here 11(i/p)+1(o/p)/2=6
#kernel_initializer how to randomly select weight
#activation the activation function
#input_dim mandatory only if initialize nural net (watch 21 video in udemy)
classifier.add(Dense(units=6, kernel_initializer = 'uniform',activation='relu',input_dim=11))

#dropout
classifier.add(Dropout(0.1))

#adding second layer
#input dim is removed because because it is required only in starting section
classifier.add(Dense(units=6, kernel_initializer = 'uniform',activation='relu'))

#dropout
classifier.add(Dropout(0.1))
# final or output layer. no of layer will depend on problem
#units = 1 because we have only on type of output that is wheather client stays or not
# actuvation function can be softmax if we have more no of outpur node
classifier.add(Dense(units=1, kernel_initializer = 'uniform',activation='sigmoid'))
# if ur logarithamatic function has more than one out come then function catergorical cross entropy
#compiling the ann
classifier.compile(optimizer ='adam',loss ='binary_crossentropy',metrics = 	['accuracy'])
#fitting ann to training set
classifier.fit(X_train,Y_train, batch_size=10,epochs = 100)

#predict the test result
Y_pred =classifier.predict(X_test)
Y_pred = (Y_pred > 0.5)
#making confusion matrix
from sklearn.metrics import  confusion_matrix
cm = confusion_matrix(Y_test,Y_pred)






