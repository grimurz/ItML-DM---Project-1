# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 00:23:47 2020

@author: white
"""

#We have used 2 different ways of doing a PCA analysis to validate the effictiveness of our results

#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------

#FIRST APPROACH: sklearn library implementation



import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pylab
from matplotlib.pyplot import (figure, title, boxplot, xticks, subplot, hist, xlabel, ylim, yticks, show, grid)
from pylab import *
from scipy.stats import zscore


#Reading the dataset data and start of the pre-processing 

heart_data = pd.read_excel('Dataset.xlsx')
 
#We need to conduct an one-out-of-K transformation to the famhist column of data into a binary 0,1 form
#that is suitable to get fed into the PCA analysis

fam_history = heart_data['famhist']
unique_hist = np.unique(fam_history)

#We have created a dictionary where the attribute Present/Absent was codified into 0,1

historyDict = dict(zip(fam_history,[1,0]))

#The transformed array replaces the famhist values

y = np.array([historyDict[cl] for cl in fam_history])
binary_heart_data= heart_data.copy()
binary_heart_data.famhist = y
binary_heart_data.drop('row.names', axis=1, inplace=True)

#Data standardization: We scale our data so that each feature has a single unit of variance.

scaler = StandardScaler()
scaler.fit(binary_heart_data)
scaled_data = scaler.transform(binary_heart_data)

#We conduct the PCA analysis


pca = PCA(n_components=10)
pca.fit(scaled_data)
explained_variance = pca.explained_variance_ratio_
print("Each component contributes in attribute variance by, \nthe 1st:",round(explained_variance[0],2)*100,"%\nthe 2nd:",round(explained_variance[1],2)*100,"%\nthe 3rd:",round(explained_variance[2],2)*100,"%\nthe 4th:",round(explained_variance[3],2)*100,"%\nthe 5th:",round(explained_variance[4],2)*100,"%")
val=0
print()
print("Variance accumulation by each PCA component:")
print()
for comp in explained_variance:
    val+=round(comp,2)*100
    print(val,"%")
   

#Now we can transform this data to its principal components


x_pca = pca.transform(scaled_data)
#We can know feed the x_pca into a classification algorithm

binary_heart_data.to_excel('binary_heart_data.xlsx')

#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------

#Second APPROACH: scipy exercise based implementation


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

#print('First 3 PC:',np.round(np.sum(rho[0:3])*100,1),'%')

threshold = 0.9


# correlation test
corr_test = np.corrcoef(X)


#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------

#PLOTTING


#First approach


classNames = ['Absent','Present'] 
plt.figure(figsize=(8,6),dpi=300)
plt.scatter(x_pca[:,2],x_pca[:,4],c=binary_heart_data.famhist,cmap='seismic',alpha=.5,sizes=(10, 70))
plt.title('Family history', fontsize=14)
plt.xlabel('Third principal component')
plt.ylabel('Fifth Principal Component')
plt.legend(['Absent','Present'])
#plt.figure(figsize=(8,6),dpi=300)
#plt.scatter(x_pca[:,2],x_pca[:,4],c=binary_heart_data.famhist,cmap=sns.cubehelix_palette(start=0,rot=-.2,light=.75,dark=.30, as_cmap=True),alpha=.7,sizes=(10, 70))
#plt.title('Family history')
#plt.xlabel('Third principal component')
#plt.ylabel('Fifth Principal Component')


#plt.figure(figsize=(8,6),dpi=300)
#plt.scatter(x_pca[:,0],x_pca[:,2],c=binary_heart_data.chd,cmap='seismic')
#plt.title('Coronary heart disease')
#plt.xlabel('First principal component')
#plt.ylabel('Third Principal Component')

#plt.figure(figsize=(8,6),dpi=300)
#plt.scatter(x_pca[:,0],x_pca[:,2],c=binary_heart_data.chd,cmap=sns.cubehelix_palette(start=2,rot=-.2,light=.75,dark=.30, as_cmap=True),alpha=.7,sizes=(10, 70))
#plt.title('Coronary heart disease')
#plt.xlabel('First principal component')
#plt.ylabel('Third Principal Component')

#This heatmap and the color bar basically represent the correlation between the various feature 
#and the principal component itself.


df_comp = pd.DataFrame(pca.components_,columns=binary_heart_data.columns)
plt.figure(figsize=(12,6),dpi=300)
sns.heatmap(df_comp,cmap='plasma',)
plt.title('Heatmap of contritbution of the attributes in the PCs', fontsize=22);
#3d scatter of the original data

fi = plt.figure(figsize=(10,5),dpi=300)
ax = fi.add_subplot(111, projection='3d')
xi = binary_heart_data.obesity
yi = binary_heart_data.tobacco
zi = binary_heart_data.alcohol
ax.scatter(xi,yi,zi, c=binary_heart_data.sbp,cmap='twilight_shifted', marker='o', alpha=.5,sizes=(10, 40))
ax.set_title('Original data color coded for systolic blood pressure', fontsize=18)
ax.set_xlabel('Obesity')
ax.set_ylabel('Tobacco use')
ax.set_zlabel('Alcohol use')

#3d scatter of the first 3 PCAs

fz = plt.figure(figsize=(10,5),dpi=300)
ax = fz.add_subplot(111, projection='3d')
xz = x_pca[:,2]
yz = x_pca[:,3]
zz = x_pca[:,4]
ax.scatter(xz,yz,zz, c=binary_heart_data.famhist,cmap='seismic', marker='o', alpha=.5,sizes=(10, 40))
ax.set_title('3d PCA representation',fontsize=18)
ax.set_xlabel('1st PC')
ax.set_ylabel('2nd PC')
ax.set_zlabel('3rd PC')


#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------

#PLOTTING


#Second approach



# Plot variance explained
plt.figure(dpi=300)
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components', fontsize=12);
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()


attributeNames = list(binary_heart_data.columns)


# Indices of the principal components to be plotted

i = 2
j = 4

# Plot PCA of the data
f = figure(dpi=300)
title('Family history', fontsize=14)
#Z = array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
legend(classNames)
xlabel('PC{0}'.format(i+1))
ylabel('PC{0}'.format(j+1))
show()


# We saw in 2.1.3 that the first 3 components explaiend more than 90
# percent of the variance. Let's look at their coefficients:
pcs = [0,1,2,3,4]
legendStrs = ['PC'+str(e+1) for e in pcs]
# c = ['r','g','b']
bw = .2
r = np.arange(1,M+1)
for i in pcs:    
    plt.bar(r+i*bw, V[:,i], width=bw*4/5)
plt.xticks(r+bw, attributeNames)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')

plt.legend(legendStrs, loc=4)


plt.xticks(rotation=22)
plt.grid()
plt.title('PCA Component Coefficients')

plt.savefig('PCA Component Coefficients.png', dpi = 300)
plt.show()

r = np.arange(1,X.shape[1]+1)
plt.bar(r, np.std(X,0))
plt.xticks(r, attributeNames)
plt.xticks(rotation=45)
plt.ylabel('Standard deviation')
plt.xlabel('Attributes')
plt.title('Attribute standard deviations')
plt.xticks(rotation=45)
plt.show()












#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------

#GENERIC STATISTICS


#Distributions - correlations


correlation = binary_heart_data[['famhist','chd']].copy()
chd_positive = binary_heart_data[['famhist','chd']].copy()
double_positive = correlation[(correlation['famhist']>0) &(correlation['chd']>0)]
hist_positive_chd_negative = correlation[(correlation['famhist']>0) &(correlation['chd']==0)]
hist_negative_chd_positive =correlation[(correlation['famhist']==0) &(correlation['chd']==1)]
double_negative=correlation[(correlation['famhist']==0) &(correlation['chd']==0)]
dn_count=len(double_negative)
hist_negative_only = len(hist_negative_chd_positive)
hist_positive_only = len(hist_positive_chd_negative)
dp_count = len(double_positive)
chd_count = len(chd_positive)
chd_positive.drop(chd_positive[chd_positive['chd']==0].index, axis=0, inplace=True)
outcomes= ["Double positive", "Double negative", "+/-", "-/+"]
length_list = [dp_count,dn_count,hist_positive_only,hist_negative_only]
lengths =np.array(length_list)
family_hist=pd.DataFrame(lengths, ["Double positive", "Double negative", "+/-", "-/+"])
family_hist.T

famhist_col = list(family_hist.index)
famhist_x = family_hist.values.T[0]
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(famhist_col,famhist_x)
plt.title('Family History/CHD correlation color-coded for Coronary Heart Disease', fontsize=14)
plt.xlabel('Family History', fontsize=14)
plt.ylabel('CHD', fontsize=14)
plt.xticks(fontsize=12)
plt.show()


attributeNames = list(binary_heart_data.columns)

X = binary_heart_data.to_numpy()

mean_X = np.round(np.mean(X,0),3)
std_X = np.round(np.std(X,0),3)
max_X = np.round(X.max(0),3)
min_X = np.round(X.min(0),3)
median_X = np.round(np.median(X,0),3)
range_X = np.round(X.max(0) - X.min(0),3)

print("The probability of CHD if family history is present is:", (dp_count/len(chd_positive))*100,"%")


nonbinary_heart_data = binary_heart_data.drop(['famhist','chd'], axis=1).to_numpy()
nonbinary_heart_attr = list(binary_heart_data.drop(['famhist','chd'], axis=1).columns)

# boxprops = dict(linewidth=3)

figure(figsize=(20,10))
# figure()
subplot(1,2,1)
title('CHD: Boxplot', fontsize=18)
boxplot(nonbinary_heart_data, 0, 'rx')
grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
M = nonbinary_heart_data.shape[1]
xticks(range(1,M+1), nonbinary_heart_attr, rotation=45, fontsize=16)
yticks(fontsize=16)


# figure(figsize=(8,8))
# figure()
subplot(1,2,2)
title('CHD: Boxplot (standarized)', fontsize=18)
boxplot(zscore(nonbinary_heart_data, ddof=1), nonbinary_heart_attr, 'rx')
grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
xticks(range(1,M+1), nonbinary_heart_attr, rotation=45, fontsize=16)
yticks(fontsize=16)


figure()
for num, att in enumerate(attributeNames):
    plt.hist(X[:,num], bins='auto') 
    plt.title("Distribution of "+str(att), fontsize=18)
    plt.show()

    
    # axs[num][num].hist(X[:,num], bins='auto')
    # axs[num].title("Distribution of "+str(att))
    
#plt.figure(figsize=(8,6),dpi=300)
#plt.scatter(heart_data.ldl,heart_data.typea,c=binary_heart_data.chd, cmap='plasma', alpha=.7, sizes=(10, 70))
#plt.title('LDL/Type A correlation color-coded for Coronary Heart Disease')
#plt.xlabel('LDL')
#plt.ylabel('Type A')


#plt.figure(figsize=(8,6),dpi=300)
#plt.scatter(heart_data.famhist,heart_data.chd, cmap='plasma',alpha=.7, sizes=(10, 70))
#plt.title('Family History/CHD correlation color-coded for Coronary Heart Disease')
#plt.xlabel('Family History')
#plt.ylabel('CHD')
    
plt.figure(figsize=(8,6),dpi=300)
plt.scatter(heart_data.age,heart_data.obesity,c=binary_heart_data.chd, cmap='seismic',alpha=.7, sizes=(10, 70))
# calc the trendline
z = np.polyfit(heart_data.age, heart_data.obesity, 1)
p = np.poly1d(z)
pylab.plot(heart_data.age,p(heart_data.age),"r")
plt.title('Age/Obesity correlation color-coded for Coronary Heart Disease', fontsize=16)
plt.xlabel('Age')
plt.ylabel('Obesity')

#plt.figure(figsize=(8,6),dpi=300)
#plt.scatter(heart_data.age,heart_data.typea,c=binary_heart_data.chd, cmap='plasma',alpha=.7, sizes=(10, 70))
#plt.title('Age/Type A correlation color-coded for Coronary Heart Disease')
#plt.xlabel('Age')
#plt.ylabel('Type A')


plt.figure(figsize=(8,6),dpi=300)
plt.scatter(heart_data.age,heart_data.adiposity,c=binary_heart_data.chd, cmap='seismic',alpha=.7, sizes=(10, 70))
# calc the trendline
z = np.polyfit(heart_data.age, heart_data.adiposity, 1)
p = np.poly1d(z)
pylab.plot(heart_data.age,p(heart_data.age),"r")

plt.title('Age/Adiposity correlation color-coded for Coronary Heart Disease', fontsize=16)
plt.xlabel('Age')
plt.ylabel('Adiposity')



sns.set(style="white")


d = pd.DataFrame(data=binary_heart_data,
                 columns=list(binary_heart_data.columns))

# Compute the correlation matrix
corr = d.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=np.bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9),dpi=300)


plt.suptitle('Correlation of attributes in the original data', fontsize=22)
#ax.set_title('Correlation of the attributes in the orginal data')

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sea =sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.4, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})