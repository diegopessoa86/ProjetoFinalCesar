# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 13:23:08 2019

@author: adans
"""
import time
import pandas as pd
import datetime

start = time.time()
print("1.Imports realizados")


#Definição de metodos

def retorna_values(dataset,coluna):
    values = dataset.iloc[:,coluna]
    return values


print("2.Definição de metodos realizado")
#Abrindo arquivos 

dfC = pd.read_csv("C:\CSCORE\CARTAO.txt", sep='\t', dtype='unicode')
dfClientes = pd.read_csv("C:\CSCORE\CLIENTES_2017.txt", sep='\t', dtype='unicode')
print("3.DataFrames Carregados")
#============================================================

#Adicionando as COLUNAS novas

dfC['FLAG_TARGET'] = 1
dfC["FLAG_ELEGIVEL"] = 1
print("4.Colunas Criadas")


dfC = dfC.merge(dfClientes, left_on='CODIGO_CLIENTE', right_on='CODIGO_CLIENTE', how='inner')
print("5.Tabelas concatenadas")

dfC = retorna_values(dfC, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18])
    

#Testando as datas para elegivel e target
progress = 1
for index, row in dfC.iterrows():
    if pd.to_datetime(row['DATA_PAGAMENTO']) <= pd.to_datetime(row['DATA_VENCIMENTO']):
        dfC.loc[index,'FLAG_TARGET'] = 0
    else:
        dfC.loc[index,'FLAG_TARGET'] = 1
    print((progress/739905)*100)
    progress = progress + 1   
    
print("6. Targets atribuidos")

dfC.to_csv("C:\CSCORE\TABELAO_CARTAO.txt", sep='\t', encoding='utf-8')
print("7.Arquivo Criado") 

end = time.time()
print(end - start)