# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 18:11:46 2019
Projeto Final do Curso Machine Learning === C.E.S.A.R School
@author: adans
"""

#Imports das Bibliotecas

import pandas as pd
import pyspark as sp
import datetime

#Definição de metodos

def retorna_values(dataset,coluna):
    values = dataset.iloc[:,coluna]
    return values



#Abrindo arquivos 

dfCadastral = pd.read_csv("C:\CSCORE\CADASTRAL.txt", sep='\t', dtype='unicode')

#============================================================

dataLimiteInferior = pd.to_datetime('2017/01/01')
dataLimiteSuperior = pd.to_datetime('2017/12/31')

df = retorna_values(dfCadastral,[0,9])
df['DATA_LIMITE'] = pd.NaT

df = df[pd.to_datetime(df.DATA_CADASTRO) >= dataLimiteInferior]
df = df[pd.to_datetime(df.DATA_CADASTRO) < dataLimiteSuperior]

df.DATA_LIMITE = pd.to_datetime(df.DATA_CADASTRO) + datetime.timedelta(days=180)

df.to_csv("C:\CSCORE\CLIENTES_2017.txt", sep='\t', encoding='utf-8')

