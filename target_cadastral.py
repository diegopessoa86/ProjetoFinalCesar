# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 10:54:07 2019

@author: adans
"""

import pandas as pd
import datetime
from datetime import date

def retorna_values(dataset,coluna):
    values = dataset.iloc[:,coluna]
    return values


#dfClientes = pd.read_csv("C:\CSCORE\CLIENTES_2017.txt", sep='\t', dtype='unicode')


# VENDOS OS TARGETS POR CLIENTE NO CARTAO
dfC = pd.read_csv("C:\CSCORE\TABELAO_CARTAO_TARGET_ELEGIVEL.txt", sep='\t', dtype='unicode')

dfC['FLAG_TARGET'] = dfC['FLAG_TARGET'].astype(int)

dfTarget_Cartao_Clientes = dfC.groupby('CODIGO_CLIENTE')['FLAG_TARGET'].sum()

dfTarget_Cartao_Clientes = dfTarget_Cartao_Clientes.to_frame()
dfTarget_Cartao_Clientes['CODIGO_CLIENTE'] = 0
dfTarget_Cartao_Clientes['CODIGO_CLIENTE'] = dfTarget_Cartao_Clientes.index


#VENDO OS TARGET POR CLIENTES NO CARNE
dfC = pd.read_csv("C:\CSCORE\TABELAO_CARNE.txt", sep='\t', dtype='unicode')

dfC['FLAG_TARGET'] = dfC['FLAG_TARGET'].astype(int)

dfTarget_Carne_Clientes = dfC.groupby('CODIGO_CLIENTE')['FLAG_TARGET'].sum()
dfTarget_Carne_Clientes = dfTarget_Carne_Clientes.to_frame()
dfTarget_Carne_Clientes['CODIGO_CLIENTE'] = 0
dfTarget_Carne_Clientes['CODIGO_CLIENTE'] = dfTarget_Carne_Clientes.index

#TABELAS CONCLUIDAS

dfTarget = dfTarget_Carne_Clientes.merge(dfTarget_Cartao_Clientes, left_on='CODIGO_CLIENTE', right_on='CODIGO_CLIENTE', how='outer')
dfTarget = dfTarget.fillna(0)

dfTarget['FLAG_TARGET'] = dfTarget['FLAG_TARGET_x'] + dfTarget['FLAG_TARGET_y']
#dfTarget['FLAG_TARGET'] = dfTarget['FLAG_TARGET_y']
#dfTarget['FLAG_TARGET'] = dfTarget['FLAG_TARGET_x'] 

#dfTarget.to_csv("C:\CSCORE\TARGET.txt", sep='\t', encoding='utf-8')

dfC = pd.read_csv("C:\CSCORE\CADASTRAL.txt", sep='\t', dtype='unicode')
dfC = dfC.merge(dfTarget, left_on='CODIGO_CLIENTE', right_on='CODIGO_CLIENTE', how='outer')

dataLimiteInferior = pd.to_datetime('2017/01/01')
dataLimiteSuperior = pd.to_datetime('2017/12/31')

dfC = dfC[pd.to_datetime(dfC.DATA_CADASTRO) >= dataLimiteInferior]
dfC = dfC[pd.to_datetime(dfC.DATA_CADASTRO) < dataLimiteSuperior]

today = pd.to_datetime(date.today() )

dfC['IDADE'] = 0

dfC['IDADE'] = today - pd.to_datetime(dfC['DATA_NASCIMENTO'])

dfC = retorna_values(dfC,[0,1,2,3,4,5,6,8,9,10,11,12,13,14,15,16,17,18,21,22])



dfC.to_csv("C:\CSCORE\TABELAO_.txt", sep='\t', encoding='utf-8')

dfC = dfC[dfC["FLAG_TARGET"] >= 0]

progress = 1
for index, row in dfC.iterrows():
    if pd.to_numeric(row['FLAG_TARGET']) > 0:
        dfC.loc[index,'FLAG_TARGET'] = 1
    print((progress/96228)*100)
    progress = progress + 1 

dfC.to_csv("C:\CSCORE\TABELAO_1.txt", sep='\t', encoding='utf-8')
