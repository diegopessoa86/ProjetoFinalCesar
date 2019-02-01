# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 12:23:33 2019

@author: adans
"""

import matplotlib.pyplot as plt 
import pandas as pd  
import statsmodels.api as sm  
import pylab as pl  
import numpy as np  
from patsy import dmatrix 

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def retorna_values(dataset,coluna):
    values = dataset.iloc[:,coluna]
    return values

dfLimite = pd.read_csv("C:\CSCORE\TABELAO_CARTAO_TARGET_ELEGIVEL.txt", sep='\t')
dfLimite = dfLimite[dfLimite["FLAG_TARGET"] >= 0]
dfLimite = dfLimite.groupby('CODIGO_CLIENTE')['LIMITE'].max()
dfLimite = dfLimite.to_frame()
dfLimite['CODIGO_CLIENTE'] = dfLimite.index
dfLimite['FLAG_CARTAO'] = 1

dfCarne = pd.read_csv("C:\CSCORE\TABELAO_CARNE.txt", sep='\t')
dfCarne = dfCarne[dfCarne["FLAG_TARGET"] >= 0]
dfCarne = dfCarne.groupby('CODIGO_CLIENTE')['VALOR_PARCELA'].mean()
dfCarne = dfCarne.to_frame()
dfCarne['CODIGO_CLIENTE'] = dfCarne.index
dfCarne['FLAG_CARNE'] = 1


df = pd.read_csv("C:\CSCORE\TABELAO_1.txt", sep='\t')
df = retorna_values(df,[1,3,4,7,8,11,12,13,14,15,16,17,18,19,20])
df = df[df["FLAG_TARGET"] >= 0]

df = df.merge(dfLimite, left_on='CODIGO_CLIENTE', right_on='CODIGO_CLIENTE', how='outer') # adicionado a coluna LIMITE
df = df.merge(dfCarne, left_on='CODIGO_CLIENTE', right_on='CODIGO_CLIENTE', how='outer')
#TRATAMENTO DE DADOS
#CONVERTENDO A IDADE EM DIAS(TIMEDELTA) EM ANOS(INT)

df['IDADE'] = pd.to_timedelta(df['IDADE'])
td = df['IDADE']
df['IDADE'] = pd.to_numeric(td.dt.days/365)
#df['ESTADO_CIVIL'] = df['ESTADO_CIVIL'].fillna('CA')

#GERANDO OS DUMMIES PARA SEXO, ESTADO_CIVIL, CATEGORIA PROFISSIONAL, TIPO DE RESIDENCIA
sexo = pd.get_dummies(df['SEXO'], drop_first=True)
estado_civil = pd.get_dummies(df['ESTADO_CIVIL'], drop_first=True)
categoria_prof = pd.get_dummies(df['CATEGORIAL_PROFISSAO'], drop_first=True)
tipo_residencia = pd.get_dummies(df['TIPO_RESIDENCIA'], drop_first=True)


#PREENCHENDO OS NAN COM 'N'
df['FLAG_CONTA_BANCO'] = df['FLAG_CONTA_BANCO'].fillna('N')
df['LIMITE'] = df['LIMITE'].fillna(0)
df['FLAG_CARTAO'] = df['FLAG_CARTAO'].fillna(0)

df['VALOR_PARCELA'] = df['VALOR_PARCELA'].fillna(0)
df['FLAG_CARNE'] = df['FLAG_CARNE'].fillna(0)

#CRIANDO DUMMIES PARA FLAG_CONTA_BANCO

flag_conta_banco = pd.get_dummies(df['FLAG_CONTA_BANCO'], drop_first = True)

#CONVERTER tudo PARA NUMERO
#TRATANDO OS NAN COM 0
#df['RENDA_CJ'] = df['RENDA_CJ'].fillna('0')
#df['RENDA_TITULAR'] = df['RENDA_TITULAR'].fillna('0')
#df['OUTRAS_RENDAS'] = df['OUTRAS_RENDAS'].fillna('0')

df = df.apply(pd.to_numeric, errors='coerce')


df = df.drop('BAIRRO', axis=1)
df = df.drop('ESTADO_CIVIL', axis=1)
df = df.drop('SEXO', axis=1)
df = df.drop('CATEGORIAL_PROFISSAO', axis=1)
df = df.drop('TIPO_RESIDENCIA', axis=1)
df = df.drop('FLAG_CONTA_BANCO', axis=1)
df = df.drop('QTD_CARTOES_ADICIONAIS', axis=1)
#CRIA O DF para o tabelao
#df = pd.concat([df, sexo, estado_civil, categoria_prof, tipo_residencia, flag_conta_banco],axis=1)

df = pd.concat([df, sexo, estado_civil, categoria_prof, tipo_residencia, flag_conta_banco],axis=1)

#print(df.dtypes)

print("DATA SET SAMPLE")
print(df.head(10))

print("DATA SUMMARY")
print(df.describe())

print("DESVIO PADRAO")
print(df.std())

from sklearn.preprocessing import Imputer

imp1 = Imputer(missing_values = 'NaN', strategy='median', axis=0)
imp2 = Imputer(missing_values = 'NaN', strategy='median', axis=0)
imp3 = Imputer(missing_values = 'NaN', strategy='median', axis=0)
imp4 = Imputer(missing_values = 'NaN', strategy='median', axis=0)
imp5 = Imputer(missing_values = 'NaN', strategy='median', axis=0)

imp1.fit(df['RENDA_CJ'].values.reshape(-1,1))
df['RENDA_CJ'] = imp1.transform(df['RENDA_CJ'].values.reshape(-1,1))

imp2.fit(df['RENDA_TITULAR'].values.reshape(-1,1))
df['RENDA_TITULAR'] = imp2.transform(df['RENDA_TITULAR'].values.reshape(-1,1))

imp3.fit(df['OUTRAS_RENDAS'].values.reshape(-1,1))
df['OUTRAS_RENDAS'] = imp3.transform(df['OUTRAS_RENDAS'].values.reshape(-1,1))

imp4.fit(df['IDADE'].values.reshape(-1,1))
df['IDADE'] = imp4.transform(df['IDADE'].values.reshape(-1,1))

imp5.fit(df['LIMITE'].values.reshape(-1,1))
df['LIMITE'] = imp5.transform(df['LIMITE'].values.reshape(-1,1))


df['RENDA_CJ'] = df['RENDA_CJ'].fillna(0)
df['RENDA_TITULAR'] = df['RENDA_TITULAR'].fillna(0)
df['OUTRAS_RENDAS'] = df['OUTRAS_RENDAS'].fillna(0)
df['IDADE'] = df['IDADE'].astype(int)
df['IDADE'] = df['IDADE'].fillna(0)
df['QTD_DEPENDENTES'] = df['QTD_DEPENDENTES'].fillna(0)

#df['RENDA'] = df['RENDA_CJ'] + df['RENDA_TITULAR'] + df['OUTRAS_RENDAS']

#df = df.drop('RENDA_CJ', axis=1)
#df = df.drop('RENDA_TITULAR', axis=1)
#df = df.drop('OUTRAS_RENDAS', axis=1)

df.to_csv("C:\CSCORE\TABELAO_MODELO_1.txt", sep='\t', encoding='utf-8')
X = df.drop('FLAG_TARGET', axis=1)
y = df['FLAG_TARGET']

sc = StandardScaler()
#X =  sc.fit_transform(X)

print(df.dtypes)
print(df.info())

import seaborn as sns
corr = df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

#dfC.to_csv("C:\CSCORE\TABELAO_1.txt", sep='\t', encoding='utf-8')

"""
==========================================================
============INICIO DO MODELO 1============================
============ REGRESSAO LOGISTICA==========================
==========================================================
"""

print("MODELO 1.")
print("MODELO REGRESSAO LOGISTICA")
print()

model = LogisticRegression()

#from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.25, random_state=63)

model.fit(X_train,y_train)

#from sklearn.metrics import classification_report

predictions = model.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print("ACCURACY:")
print(accuracy_score(y_test,predictions))
print()
"""
==========================================================
============INICIO DO MODELO 2============================
============ REDES NEURAIS================================
==========================================================
"""
print("MODELO 2.")
print("MODELO REDES NEURAIS")
print()

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)

mlp.fit(X_train,y_train)

predictions = mlp.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print("ACCURACY:")
print(accuracy_score(y_test,predictions))
print()


from sklearn.model_selection import cross_val_score

X = df.drop('FLAG_TARGET', axis=1)
y = df['FLAG_TARGET']
"""
model = MLPClassifier(hidden_layer_sizes=(100,),max_iter=2500)
scores = cross_val_score(model, X, y, cv=10)
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
"""

"""
==========================================================
============INICIO DO MODELO 3============================
================= ARVORE =================================
==========================================================
"""
print("MODELO 3.")
print("MODELO ARVORE")
print()
from sklearn import tree

model = tree.DecisionTreeClassifier() # Cria árvore de classificação
model = tree.DecisionTreeRegressor() # Cria árvore de regressão

model.fit(X_train,y_train) # Cria a árvore baseada nos dados de treinamento
predictions = model.predict(X_test) # Prediz a saída dos dados de teste

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print("ACCURACY:")
print(accuracy_score(y_test,predictions))
print()
"""
==========================================================
============INICIO DO MODELO 4============================
================= RANDOM FOREST===========================
==========================================================
"""
from sklearn import ensemble # importa o pacote ensemble

# Cria o modelo com 100 árvores
model = ensemble.RandomForestClassifier ( n_estimators=10 )
model = ensemble.RandomForestRegressor ( n_estimators=10 )
model.fit ( X_train, y_train ) # Treina o modelo

predictions = model.predict(X_test) # Utiliza o modelo na base de testes

print(accuracy_score(y_test,predictions))

print(confusion_matrix(y_test,predictions))
#print(classification_report(y_test,predictions))

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

rfe = RFE(logreg, 20)
rfe = rfe.fit(os_data_X, os_data_y.values.ravel())
print(rfe.support_)
print(rfe.ranking_)
