# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 21:54:03 2020

@author: white
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

heart_data = pd.read_excel('Dataset.xlsx')
fam_history = heart_data['famhist']
unique_hist = np.unique(fam_history)
historyDict = dict(zip(fam_history,[0,1]))
y = np.array([historyDict[cl] for cl in fam_history])
binary_heart_data= heart_data.copy()
binary_heart_data.famhist = y    
binary_heart_data.drop('row.names', axis=1, inplace=True)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(binary_heart_data)
scaled_data = scaler.transform(binary_heart_data)

from sklearn.decomposition import PCA
pca = PCA(n_components=5)
pca.fit(scaled_data)

x_pca = pca.transform(scaled_data)


plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,2],x_pca[:,4],c=binary_heart_data.famhist,cmap='plasma')
plt.xlabel('Third principal component')
plt.ylabel('Fifth Principal Component')

df_comp = pd.DataFrame(pca.components_,columns=binary_heart_data.columns)
plt.figure(figsize=(12,6))
sns.heatmap(df_comp,cmap='plasma',)
plt.show()