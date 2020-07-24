# -*- coding: utf-8 -*-
"""
Created on Fri May 22 15:01:11 2020

@author: Renu
"""
import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Adataes3D
from sklearn.cluster import KMeans
import seaborn as sns

data = pd.read_csv("D:Data//demo_datasets_shopping.csv");
data.head();
data.drop(["CustomerID"], adatais = 1, inplace=True)
#Changing the name of some columns
data = data.rename(columns={'Annual_income': 'Annual_income', 'Spending_score': 'Spending_score'})
#Looking for null values
data.isna().sum()
#Checking datatypes
data.info()
#Replacing objects for numerical values
data['Gender'].replace(['Female','Male'], [0,1],inplace=True)
data.Gender
#Density estimation of values using distplot
plt.figure(1 , figsize = (15 , 6))
feature_list = ['Age','Annual_income', "Spending_score"]
feature_listt = ['Age','Annual_income', "Spending_score"]
pos = 1 
for i in feature_list:
    plt.subplot(1 , 3 , pos)
    plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
    sns.distplot(data[i], bins=20, kde = True)
    pos = pos + 1
plt.show()
#Count and plot gender
sns.countplot(y = 'Gender', data = data, palette="husl", hue = "Gender")
data["Gender"].value_counts()
#Pairplot with variables we want to study
sns.pairplot(data, vars=["Age", "Annual_income", "Spending_score"],  kind ="reg", hue = "Gender", palette="husl", markers = ['o','D'])
#age and annual income
sns.lmplot(data = "Age", y = "Annual_income", data = data, hue = "Gender")
#spending score and annual income
sns.lmplot(data = "Annual_income", y = "Spending_score", data = data, hue = "Gender")
#age and spending score
sns.lmplot(data = "Age", y = "Spending_score", data = data, hue = "Gender")

#clusters
K=5
# Select random observation as centroids
Centroids = (data.sample(n=K))
plt.scatter(data["Annual_income"],Y["Spending_score"],c='black')
plt.scatter(Centroids["Annual_income"],Centroids["Spending_score"],c='red')
plt.datalabel('AnnualIncome')
plt.ylabel('Spending_score')
plt.show()

diff = 1
j=0

while(diff!=0):
    dataD=data
    i=1
    for indedata1,row_c in Centroids.iterrows():
        ED=[]
        for indedata2,row_d in dataD.iterrows():
            d1=(row_c["Annual_income"]-row_d["Annual_income"])**2
            d2=(row_c["Spending_score"]-row_d["Spending_score"])**2
            d=np.sqrt(d1+d2)
            ED.append(d)
        data[i]=ED
        i=i+1

    C=[]
    for indedata,row in data.iterrows():
        min_dist=row[1]
        pos=1
        for i in range(K):
            if row[i+1] < min_dist:
                min_dist = row[i+1]
                pos=i+1
        C.append(pos)
    data["Cluster"]=C
    Centroids_new = data.groupby(["Cluster"]).mean()[["Spending_score","Annual_income"]]
    if j == 0:
        diff=1
        j=j+1
    else:
        diff = (Centroids_new['Spending_score'] - Centroids['Spending_score']).sum() + (Centroids_new['Annual_income'] - Centroids['Annual_income']).sum()
        print(diff.sum())
    Centroids = data.groupby(["Cluster"]).mean()[["Spending_score","Annual_income"]]
    
#3d distribution
sns.set_style("white")
fig = plt.figure(figsize=(10,5))
adata = fig.add_subplot(111, projection='3d')
adata.scatter(data.Age, data["Annual_income"], data["Spending_score"], c='blue', s=60)
adata.view_init(30, 185)
plt.datalabel("Age")
plt.ylabel("Annual_income")
adata.set_zlabel('Spending_score')
plt.show()


#elbow method
wcss = []
for k in range(1,11):
    kmeans = KMeans(n_clusters=k, init="k-means++")
    kmeans.fit(data.iloc[:,1:])
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(12,6))    
plt.grid()
plt.plot(range(1,11),wcss, linewidth=2, color="red", marker ="8")
plt.datalabel("K Value")
plt.dataticks(np.arange(1,11,1))
plt.ylabel("WCSS")
plt.show()

km = KMeans(n_clusters=6)
clusters = km.fit_predict(data.iloc[:,1:])
data["label"] = clusters
#final segmentation
fig = plt.figure(figsize=(10,5))
adata = fig.add_subplot(111, projection='3d')
adata.scatter(data.Age[data.label == 0], data["Annual_income"][data.label == 0], data["Spending_score"][data.label == 0], c='red', s=60)
adata.scatter(data.Age[data.label == 1], data["Annual_income"][data.label == 1], data["Spending_score"][data.label == 1], c='black', s=60)
adata.scatter(data.Age[data.label == 2], data["Annual_income"][data.label == 2], data["Spending_score"][data.label == 2], c='green', s=60)
adata.scatter(data.Age[data.label == 3], data["Annual_income"][data.label == 3], data["Spending_score"][data.label == 3], c='orange', s=60)
adata.scatter(data.Age[data.label == 4], data["Annual_income"][data.label == 4], data["Spending_score"][data.label == 4], c='blue', s=60)
adata.view_init(30, 185)
plt.datalabel("Age")
plt.ylabel("Annual_income")
adata.set_zlabel('Spending_score')
plt.show()







    



    

