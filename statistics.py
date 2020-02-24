
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


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


for num, att in enumerate(attributeNames):
    plt.hist(X[:,num], bins='auto') 
    plt.title("Distribution of "+str(att), fontsize=18)
    plt.show()
