# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 22:09:39 2018

@author: 15406
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import seaborn as sns; sns.set(color_codes=True)


data = pd.read_csv('22.csv')
data5 = data['5_star'].values
data4 = data['4_star'].values
data3 = data['3_star'].values
data2 = data['2_star'].values
data1 = data['1_star'].values
dataf = data['fake_review_proportion'].values
data_s = data['rating'].values
data_s = data_s/2

data_st = (5-data_s)**2*data5 + (4-data_s)**2*data4 + (3-data_s)**2*data3 + (2-data_s)**2*data2 + (1-data_s)**2*data1 

data_sst5 = np.nan_to_num(((5-data_s)/(abs(5-data_s)))*(5-data_s)**2*data5)
data_sst4 = np.nan_to_num(((4-data_s)/(abs(4-data_s)))*(4-data_s)**2*data4)
data_sst3 = np.nan_to_num(((3-data_s)/(abs(3-data_s)))*(3-data_s)**2*data3)
data_sst2 = np.nan_to_num(((2-data_s)/(abs(2-data_s)))*(2-data_s)**2*data2)
data_sst1 = np.nan_to_num(((1-data_s)/(abs(1-data_s)))*(1-data_s)**2*data1)
data_sst = data_sst5 + data_sst4 + data_sst3 + data_sst2 + data_sst1 

data_s_mean = np.mean(data_s)
data_s_std = np.std(data_s)
scaled_data_s = np.zeros((len(data_s), ))
for i in range(len(data_s)):
    scaled_data_s[i] = (data_s[i] - data_s_mean)/data_s_std
    
data_st_mean = np.mean(data_st)
data_st_std = np.std(data_st)
scaled_data_st = np.zeros((len(data_st), ))
for i in range(len(data_st)):
    scaled_data_st[i] = (data_st[i] - data_st_mean)/data_st_std
    
data_sst_mean = np.mean(data_sst)
data_sst_std = np.std(data_sst)
scaled_data_sst = np.zeros((len(data_sst), ))
for i in range(len(data_sst)):
    scaled_data_sst[i] = (data_sst[i] - data_sst_mean)/data_sst_std

X = np.concatenate(([scaled_data_s],[scaled_data_st],[scaled_data_sst]),axis=0).T
Y = dataf
LR = LinearRegression()
train_phase = LR.fit(X,Y)
y_pred = train_phase.predict(X)
print(y_pred)

ax = sns.regplot(y=y_pred, x=data_s, marker='.', order=1, truncate=True, ci=100)
ax.set_ylim(-0.5, 1.5)
plt.title('trust score')
plt.xlabel('Average rating')
plt.ylabel('Ratio of fake reviews')
plt.savefig('/Desktop/fig5.png')
plt.show()
