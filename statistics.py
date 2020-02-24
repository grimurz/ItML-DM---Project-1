
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import pylab
from pylab import *

heart_data = pd.read_excel('Dataset.xlsx')
fam_history = heart_data['famhist']
unique_hist = np.unique(fam_history)
historyDict = dict(zip(fam_history,[1,0]))
y = np.array([historyDict[cl] for cl in fam_history])
binary_heart_data = heart_data.copy()
binary_heart_data.famhist = y
binary_heart_data.drop('row.names', axis=1, inplace=True)

attributeNames = list(binary_heart_data.columns)

X = binary_heart_data.to_numpy()

mean_X = np.round(np.mean(X,0),3)
std_X = np.round(np.std(X,0),3)
median_X = np.round(np.median(X,0),3)
range_X = np.round(X.max(0) - X.min(0),3)


# fig, axs = plt.subplots(2, 5, sharey=True, tight_layout=True)

for num, att in enumerate(attributeNames):
    plt.hist(X[:,num], bins='auto') 
    plt.title("Distribution of "+str(att))
    plt.show()
    
    # axs[num][num].hist(X[:,num], bins='auto')
    # axs[num].title("Distribution of "+str(att))
    
plt.figure(figsize=(8,6),dpi=300)
plt.scatter(heart_data.ldl,heart_data.typea,c=binary_heart_data.chd, cmap='plasma', alpha=.7, sizes=(10, 70))
plt.title('LDL/Type A correlation color-coded for Coronary Heart Disease')
plt.xlabel('LDL')
plt.ylabel('Type A')

plt.figure(figsize=(8,6),dpi=300)
plt.scatter(heart_data.obesity,heart_data.typea,c=binary_heart_data.chd, cmap='plasma',alpha=.7, sizes=(10, 70))
plt.title('Obesity/Type A correlation color-coded for Coronary Heart Disease')
plt.xlabel('Obesity')
plt.ylabel('Type A')
    
plt.figure(figsize=(8,6),dpi=300)
plt.scatter(heart_data.age,heart_data.obesity,c=binary_heart_data.chd, cmap='plasma',alpha=.7, sizes=(10, 70))
# calc the trendline
z = np.polyfit(heart_data.age, heart_data.obesity, 1)
p = np.poly1d(z)
pylab.plot(heart_data.age,p(heart_data.age),"r")
plt.title('Age/Obesity correlation color-coded for Coronary Heart Disease')
plt.xlabel('Age')
plt.ylabel('Obesity')

plt.figure(figsize=(8,6),dpi=300)
plt.scatter(heart_data.age,heart_data.typea,c=binary_heart_data.chd, cmap='plasma',alpha=.7, sizes=(10, 70))
plt.title('Age/Type A correlation color-coded for Coronary Heart Disease')
plt.xlabel('Age')
plt.ylabel('Type A')


plt.figure(figsize=(8,6),dpi=300)
plt.scatter(heart_data.age,heart_data.adiposity,c=binary_heart_data.chd, cmap='plasma',alpha=.7, sizes=(10, 70))
# calc the trendline
z = np.polyfit(heart_data.age, heart_data.adiposity, 1)
p = np.poly1d(z)
pylab.plot(heart_data.age,p(heart_data.age),"r")

plt.title('Age/Adiposity correlation color-coded for Coronary Heart Disease')
plt.xlabel('Age')
plt.ylabel('Adiposity')