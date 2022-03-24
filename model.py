
### Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle

warnings.filterwarnings('ignore')

### Import Datset
df = pd.read_csv("data_breast-cancer-wiscons.csv")
# we change the class values (at the column number 2) from B to 0 and from M to 1
df.iloc[:,1].replace('B', 0,inplace=True)
df.iloc[:,1].replace('M', 1,inplace=True)

### Splitting Data
# texture_mean','area_mean','concavity_mean','area_se','concavity_se','fractal_dimension_se',
# 'smoothness_worst','concavity_worst', 'symmetry_worst','fractal_dimension_worst']
X = df[['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
       'concavity_mean', 'concave points_mean', 'area_se', 'radius_worst',
       'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
       'concavity_worst', 'concave points_worst', 'symmetry_worst']]
y = df['diagnosis']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=0)

#### Data Preprocessing

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x_train = scaler.fit_transform(X_train)
x_test = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression
##
#import keras
#from keras.models import Sequential
#from keras.layers import Dense, Dropout

#classifier = Sequential()
#classifier.add(Dense(16, activation='relu', input_dim=15))
#classifier.add(Dropout(rate=0.1))
#classifier.add(Dense(1, activation='sigmoid'))
#classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#classifier.fit(X_train, y_train, batch_size=75, epochs=120)
#y_pred = classifier.predict(X_test)
#y_pred = (y_pred > 0.5)
clf_lr = LogisticRegression()
clf_lr.fit(x_train, y_train)
y_pred = clf_lr.predict(x_test)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

print("Confusion Matrix : \n\n" , confusion_matrix(y_pred,y_test))

print("Classification Report : \n\n" , classification_report(y_pred,y_test),"\n")


pickle.dump(clf_lr, open('model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))
print(model)
'''
from sklearn.model_selection import train_test_split



#### Data Preprocessing

from sklearn.preprocessing import LabelEncoder
labelencoder_X_1 = LabelEncoder()
y = labelencoder_X_1.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=0)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x_train = scaler.fit_transform(X_train)
x_test = scaler.transform(X_test)




##
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

classifier = Sequential()
classifier.add(Dense(16, activation='relu', input_dim=15))
classifier.add(Dense(16, activation='relu'))
classifier.add(Dropout(rate=0.1))
classifier.add(Dense(1, activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
classifier.fit(X_train, y_train, batch_size=100, epochs=600)
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
# clf_lr = LogisticRegression()
# clf_lr.fit(x_train, y_train)
# predictions = clf_lr.predict(x_test)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

print("Confusion Matrix : \n\n" , confusion_matrix(y_pred,y_test))

print("Classification Report : \n\n" , classification_report(y_pred,y_test),"\n")


pickle.dump(classifier, open('model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))
print(model)
'''
