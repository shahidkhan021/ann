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

# import keras
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import Dropout
# from keras.wrappers.scikit_learn import KerasClassifier
# from sklearn.model_selection import cross_val_score
# def build_classifier():
# 	classifier = Sequential()
# 	classifier.add(Dense(units=6, kernel_initializer = 'uniform',activation='relu',input_dim=11))
# 	classifier.add(Dense(units=6, kernel_initializer = 'uniform',activation='relu'))
# 	classifier.add(Dense(units=1, kernel_initializer = 'uniform',activation='sigmoid'))
# 	classifier.compile(optimizer ='adam',loss ='binary_crossentropy',metrics = 	['accuracy'])
# 	return classifier

# classifier = KerasClassifier(build_fn=build_classifier, batch_size=10,epochs = 100)
# accuracies = cross_val_score(estimator = classifier,X= X_train,y= Y_train,cv=10, n_jobs = 1)
# mean = accuracies.mean()
# variance = accuracies.std()


# we are doing this for check real relevance accuracy is  closer
# to first one we obtain this is rather 86 percent
# second one we obtain that is rather 84 percent and then second thing we want to check is
# where we are in the device there and straight off


#implementing drop out

#tuning the ann
import keras
from keras.models import Sequential
from keras.layers import Dense		
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
def build_classifier(optimizer):
	classifier = Sequential()
	classifier.add(Dense(units=6, kernel_initializer = 'uniform',activation='relu',input_dim=11))
	classifier.add(Dense(units=6, kernel_initializer = 'uniform',activation='relu'))
	classifier.add(Dense(units=1, kernel_initializer = 'uniform',activation='sigmoid'))
	classifier.compile(optimizer =optimizer,loss ='binary_crossentropy',metrics = 	['accuracy'])
	return classifier

classifier = KerasClassifier(build_fn=build_classifier)
parameters = {'batch_size':[25,32],
			  'nb_epoch':[100,500],
			  'optimizer':['adam','rmsprop']}

grid_search = GridSearchCV(estimator = classifier,
						   param_grid = parameters,
						   scoring = 'accuracy',
						   cv = 10)

grid_search = grid_search.fit(X_train,Y_train)
best_parameters = grid_search.best_params_
best_accuracy   = grid_search.best_score_

print(best_parameters)
print(best_accuracy)

