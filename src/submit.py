# -*- coding: utf-8 -*-

################## Fichier source pour générer le fichier à soumettre ##################


#Imports
import math
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV



#Load data
train = pd.read_csv('../data/labeled_dataset.csv', sep = ',', index_col = 'index')
test = pd.read_csv('../data/scoring_dataset.csv', sep = ',', index_col = 'index')
#Merging des bases de train et de test pour étudier les données dans leur ensemble.
data=pd.concat([train,test])
data=data.reset_index().set_index('index')
#Copie
data_raw=data.copy(deep=True)
data_raw = pd.get_dummies(data_raw, columns=['Categorie socio professionnelle',
                                               'Type de vehicule'],drop_first=True)

#Nous régressons sur les variables contenant des NaN
test = data_raw
l=['Age','Prime mensuelle']
for element in l:
    temp = test.copy(deep=True)
    del temp['Benefice net annuel']
    del temp['Marque']
    for e in [x for x in l if x!=element]:
        del temp[e]
    columns1 = list(temp.columns)
    columns1.remove(element)
    #retrouver les indices des lignes avec des nan pour la variable en question
    index = data_raw[element].index[data_raw[element].apply(np.isnan)]
    #drop les lignes
    temp = temp.drop(temp.index[index])
    lr = LinearRegression()
    lr.fit(temp[columns1], temp[element])

    for idx in index:
        X= data_raw.iloc[idx][columns1]
        y_lr = lr.predict(X.values.reshape(1, -1))
        #print(y_lr[0])
        data_raw.ix[idx,element]=y_lr[0]
    print('Régression achevée pour la variable : %s.') %(element)

#Véréfication et traitement du champs Marque
for element in data_raw.isnull().sum().index:
    if data_raw[element].isnull().sum()==0 and element=='Age' and element!='Benefice net annuel':
        #data_raw01[element] = data_raw01[element].fillna(data_raw01[element].median())
        print("L'age ne comporte plus de champs vides.")
    if data_raw[element].isnull().sum()==0 and element=='Prime mensuelle' and element!='Benefice net annuel':
        #data_raw01[element] = data_raw01[element].fillna(data_raw01[element].median())
        print("La Prime mensuelle ne comporte plus de champs vides.")
    if data_raw[element].dtype==np.object:
        data_raw[element]=data_raw[element].fillna(data_raw[element].describe().top)#remplacer par sa valeur top

                                  
data_raw = pd.get_dummies(data_raw, columns=['Marque'],drop_first=True)  
                                  
#Retraitement de la variable age
def traiter_age(x):
    if x<18:
        return 18
    if x>99:
        return 99
    return x
                               
data_raw['Age'] = data_raw['Age'].apply(lambda x: traiter_age(x))        

#Split la base train et test                                  
def simple_split(X):
    to_drop=['Benefice net annuel'] #on enlève la cible
    T=X[1000:]
    X=X[0:1000]  
    return X.drop(to_drop,axis=1),X['Benefice net annuel'],T.drop(to_drop,axis=1)
                                  
X,y,T=simple_split(data_raw.copy())       
                                  
                                  
#Submit
path='../models/test/'   
def write_to_file(X_prod,y_pred,name_file,path) :
    if not os.path.exists(path):
        os.makedirs(path)
    X_prod['Benefice net annuel predit'] = y_pred
    X_prod[['Benefice net annuel predit']].to_csv(path+name_file, index=True, sep='|')
    print('Ecriture finie.')
    return

#Compute                                  
lr = LinearRegression() #init
polynomial_features = PolynomialFeatures(degree=2, include_bias=False)
# Création du pipeline
pipe = Pipeline([("polynomial_features", polynomial_features),
        ("linear_regression", lr)])
grid = dict(linear_regression__normalize=[False,True]) #espace de paramètre de la régression
                                                        # choix sur la normalisation
model = GridSearchCV(pipe,param_grid=grid,cv=8)
model.fit(X, y)
                                  
#predict
y_lr = model.predict(T)
write_to_file(T.copy(), y_lr,'ma_prediction_RL.csv',path)
                              