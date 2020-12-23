# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 20:57:07 2020

@author: user
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import ensemble, metrics

#將資料讀取出來
df = pd.read_csv('HW2data.csv')
print(df.shape)
df.head()

df.info()

df['workclass'] = df['workclass'].astype('category').cat.codes
df['education'] = df['education'].astype('category').cat.codes
df['marital_status'] = df['marital_status'].astype('category').cat.codes
df['occupation'] = df['occupation'].astype('category').cat.codes
df['relationship'] = df['relationship'].astype('category').cat.codes
df['race'] = df['race'].astype('category').cat.codes
df['sex'] = df['sex'].astype('category').cat.codes
df['native_country'] = df['native_country'].astype('category').cat.codes
df['income'] = df['income'].astype('category').cat.codes

#income
plt.subplot(2,1,2)
sns.countplot(df['income'])
plt.title('income')
plt.show()

def K_fold_CV(k, data, target):  
    
    #設定subset size 即data長度/k
    subset_size = len(data)/k
    
    #設定Accuracy初始值
    Accuracy = 0
    
    for i in range(k):
        
        #切分train set和test set
        start = int(i*subset_size)
        end = int((i+1)*subset_size)      
        
        test_df = df.iloc[start:end]
        train_df = df.drop(df.index[[start,end-1]])
        
        train_y = train_df[target]
        train_X = train_df.drop([target],axis = 1)
        test_y = test_df[target]
        test_X = test_df.drop([target],axis = 1)
        
        # 建立 random forest 模型
        forest = ensemble.RandomForestClassifier(n_estimators = 100)
        forest_fit = forest.fit(train_X, train_y)

        # 預測
        test_y_predicted = forest.predict(test_X)

        # 績效
        accuracy = metrics.accuracy_score(test_y, test_y_predicted)
        print("Model", i+1," accuracy: ", accuracy)
        Accuracy = Accuracy + accuracy
        
    return Accuracy/k

k = 10
target ="income"

print("Accuracy: ",K_fold_CV(k, df, target))
#K_fold_CV(k, df)