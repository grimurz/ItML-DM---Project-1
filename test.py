# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 00:23:47 2020

@author: white
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


heart_data = pd.read_excel('Dataset.xlsx')
fam_history = heart_data['famhist']
unique_hist = np.unique(fam_history)
historyDict = dict(zip(fam_history,[1,0]))
y = np.array([historyDict[cl] for cl in fam_history])
binary_heart_data = heart_data.copy()
binary_heart_data.famhist = y
binary_heart_data.drop('row.names', axis=1, inplace=True)

scaler = StandardScaler()
scaler.fit(binary_heart_data)
scaled_data = scaler.transform(binary_heart_data)

pca = PCA(n_components=5)
pca.fit(scaled_data)

x_pca = pca.transform(scaled_data)
#We can know feed the x_pca into a classification algorithm



### TEST STUFF BELOW ###

from scipy.linalg import svd
from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show, legend


# X = binary_heart_data.to_numpy()[:,0:-1]
# y2 = binary_heart_data.to_numpy()[:,-1]
X = binary_heart_data.to_numpy()
y2 = binary_heart_data.to_numpy()[:,4]

# Compute values of N, M and C.
N = len(y)
M = X.shape[1]
C = 2



# Subtract the mean from the data and divide by the attribute standard
# deviation to obtain a standardized dataset:
Y = X - np.ones((N, 1))*X.mean(0)
Y = Y*(1/np.std(Y,0))
# Y = X - np.ones((N,1))*X.mean(axis=0)

# PCA by computing SVD of Y
U,S,Vh = svd(Y,full_matrices=False)
# scipy.linalg.svd returns "Vh", which is the Hermitian (transpose)
# of the vector V. So, for us to obtain the correct V, we transpose:
V = Vh.T  

# Project the centered data onto principal component space
Z = Y @ V

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 

print('First 3 PC:',np.round(np.sum(rho[0:3])*100,1),'%')

threshold = 0.9

# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()


attributeNames = list(binary_heart_data.columns)
classNames = ['Beep','Boop'] # Maybe flipped?

# Indices of the principal components to be plotted
i = 2
j = 4

# Plot PCA of the data
f = figure()
title('NanoNose data: PCA')
#Z = array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
legend(classNames)
xlabel('PC{0}'.format(i+1))
ylabel('PC{0}'.format(j+1))

# Output result to screen
show()


# We saw in 2.1.3 that the first 3 components explaiend more than 90
# percent of the variance. Let's look at their coefficients:
pcs = [0,1,2]
legendStrs = ['PC'+str(e+1) for e in pcs]
c = ['r','g','b']
bw = .2
r = np.arange(1,M+1)
for i in pcs:    
    plt.bar(r+i*bw, V[:,i], width=bw)
plt.xticks(r+bw, attributeNames)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('NanoNose: PCA Component Coefficients')
plt.show()



r = np.arange(1,X.shape[1]+1)
plt.bar(r, np.std(X,0))
plt.xticks(r, attributeNames)
plt.ylabel('Standard deviation')
plt.xlabel('Attributes')
plt.title('NanoNose: attribute standard deviations')




# plt.figure(figsize=(8,6),dpi=300)
# plt.scatter(x_pca[:,2],x_pca[:,4],c=binary_heart_data.famhist,cmap='plasma')
# plt.title('Family history')
# plt.xlabel('Third principal component')
# plt.ylabel('Fifth Principal Component')

# plt.figure(figsize=(8,6),dpi=300)
# plt.scatter(x_pca[:,2],x_pca[:,4],c=binary_heart_data.famhist,cmap=sns.cubehelix_palette(start=0,rot=-.2,light=.75,dark=.30, as_cmap=True),alpha=.7,sizes=(10, 70))
# plt.title('Family history')
# plt.xlabel('Third principal component')
# plt.ylabel('Fifth Principal Component')


# plt.figure(figsize=(8,6),dpi=300)
# plt.scatter(x_pca[:,0],x_pca[:,2],c=binary_heart_data.chd,cmap='plasma')
# plt.title('Coronary heart disease')
# plt.xlabel('First principal component')
# plt.ylabel('Third Principal Component')

# plt.figure(figsize=(8,6),dpi=300)
# plt.scatter(x_pca[:,0],x_pca[:,2],c=binary_heart_data.chd,cmap=sns.cubehelix_palette(start=2,rot=-.2,light=.75,dark=.30, as_cmap=True),alpha=.7,sizes=(10, 70))
# plt.title('Coronary heart disease')
# plt.xlabel('First principal component')
# plt.ylabel('Third Principal Component')


# df_comp = pd.DataFrame(pca.components_,columns=binary_heart_data.columns)
# plt.figure(figsize=(12,6),dpi=300)
# sns.heatmap(df_comp,cmap='plasma',)
# plt.show()