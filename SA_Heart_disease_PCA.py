# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 19:43:58 2020

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
plt.xlabel('Fourth principal component')
plt.ylabel('Fifth Principal Component')

plt.show()

#