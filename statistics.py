
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
attributeNames = list(binary_heart_data.columns)

X = binary_heart_data.to_numpy()

mean_X = np.round(np.mean(X,0),3)
std_X = np.round(np.std(X,0),3)
median_X = np.round(np.median(X,0),3)
range_X = np.round(X.max(0) - X.min(0),3)

plt.hist(family_hist.values)
plt.xlabel(outcomes)
plt.title("Likelihood of CHD if family history present" )
plt.show()
print("The probability of CHD if family history is present is:", (dp_count/len(chd_positive))*100,"%")



for num, att in enumerate(attributeNames):
    plt.hist(X[:,num], bins='auto') 
    plt.title("Distribution of "+str(att), fontsize=18)
    plt.show()

    
    # axs[num][num].hist(X[:,num], bins='auto')
    # axs[num].title("Distribution of "+str(att))
    
plt.figure(figsize=(8,6),dpi=300)
plt.scatter(heart_data.ldl,heart_data.typea,c=binary_heart_data.chd, cmap='plasma', alpha=.7, sizes=(10, 70))
plt.title('LDL/Type A correlation color-coded for Coronary Heart Disease')
plt.xlabel('LDL')
plt.ylabel('Type A')


#plt.figure(figsize=(8,6),dpi=300)
#plt.scatter(heart_data.famhist,heart_data.chd, cmap='plasma',alpha=.7, sizes=(10, 70))
#plt.title('Family History/CHD correlation color-coded for Coronary Heart Disease')
#plt.xlabel('Family History')
#plt.ylabel('CHD')
    
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

