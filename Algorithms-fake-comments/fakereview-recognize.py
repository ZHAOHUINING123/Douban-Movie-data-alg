# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 15:49:10 2018

@author: 15406
"""
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn.metrics import classification_report



from sklearn.model_selection import train_test_split
data = pd.read_excel('11.xlsx',header=None)

target_names = ['true reviews', 'fake reviews']

data_string = data[[8]]
data_string = data_string.values

string_length = np.zeros((len(data_string), ))
for i in range(len(data_string)):
    string_length[i] = len(data_string[i][0])

data_copy = data[[10,12,14,16,17,18,19]]
data_X = data_copy.values

string_length_mean = np.mean(string_length)
string_length_std = np.std(string_length)
scaled_string_length = np.zeros((len(data_string), ))
for i in range(len(data_string)):
    scaled_string_length[i] = (len(data_string[i][0])-string_length_mean)/string_length_std

liked_number = data_X[:,0]/data_X[:,6]
liked_number_mean = np.mean(liked_number)
liked_number_std = np.std(liked_number)
scaled_liked_number = np.zeros((len(liked_number), ))
for i in range(len(liked_number)):
    scaled_liked_number[i] = (liked_number[i]-liked_number_mean)/liked_number_std
    
comment_time =  data_X[:,3]
comment_time_mean = np.mean(comment_time)
comment_time_std = np.std(comment_time)
scaled_comment_time = np.zeros((len(comment_time), ))
for i in range(len(comment_time)):
    scaled_comment_time[i] = (comment_time[i]-comment_time_mean)/comment_time_std


score_deviation = abs(data_X[:,1] - data_X[:,2])
score_deviation_mean = np.mean(score_deviation)
score_deviation_std = np.std(score_deviation)
scaled_score_deviation = np.zeros((len(score_deviation), ))
for i in range(len(score_deviation)):
    scaled_score_deviation[i] = (score_deviation[i]-score_deviation_mean)/score_deviation_std


sync = data_X[:,4]/data_X[:,5]
sync_mean = np.mean(sync)
sync_std = np.std(sync)
scaled_sync = np.zeros((len(sync), ))
for i in range(len(sync)):
    scaled_sync[i] = (sync[i]-sync_mean)/sync_std

fake = data[[13]]
fake_y = fake.values

Y = fake_y
X = np.column_stack((scaled_string_length,scaled_liked_number,scaled_comment_time,scaled_score_deviation,scaled_sync))
clf = GaussianNB()
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)
train_phase = clf.fit(X_train,y_train.ravel())
y_pred = train_phase.predict(X_test)

k = 0
j=0
n=0
m=0

for i in range(0,1000):
    if (y_pred[i] != y_test[i]):
        if y_pred[i] == 1:
            k = k+1
        else:
            j = j+1
    else:
        if y_pred[i] == 1:
            n = n+1
        else:
            m=m+1
print(k,j,n,m)
print(classification_report(y_test, y_pred, target_names=target_names))
        


clt = SVC()
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)
train_phase = clt.fit(X_train,y_train.ravel())
y_pred = train_phase.predict(X_test)

k = 0
j=0
n=0
m=0

for i in range(0,1000):
    if (y_pred[i] != y_test[i]):
        if y_pred[i] == 1:
            k = k+1
        else:
            j = j+1
    else:
        if y_pred[i] == 1:
            n = n+1
        else:
            m=m+1
print(k,j,n,m)
print(classification_report(y_test, y_pred, target_names=target_names))
clk = KNeighborsClassifier()
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)
train_phase = clk.fit(X_train,y_train.ravel())
y_pred = train_phase.predict(X_test)

k = 0
j=0
n=0
m=0

for i in range(0,1000):
    if (y_pred[i] != y_test[i]):
        if y_pred[i] == 1:
            k = k+1
        else:
            j = j+1
    else:
        if y_pred[i] == 1:
            n = n+1
        else:
            m=m+1
print(k,j,n,m)
print(classification_report(y_test, y_pred, target_names=target_names))
cld = tree.DecisionTreeClassifier()
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)
train_phase = cld.fit(X_train,y_train.ravel())
y_pred = train_phase.predict(X_test)

k = 0
j=0
n=0
m=0

for i in range(0,1000):
    if (y_pred[i] != y_test[i]):
        if y_pred[i] == 1:
            k = k+1
        else:
            j = j+1
    else:
        if y_pred[i] == 1:
            n = n+1
        else:
            m=m+1
print(k,j,n,m)
print(classification_report(y_test, y_pred, target_names=target_names))



