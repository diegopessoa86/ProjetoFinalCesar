# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 20:52:21 2019

@author: adans
"""
#NOTA
"""
Decidi usar a data de vencimento porque ele é 1 parametro melhor para separar os elegiveis
pois há casos que a data de pagamento é nula e está dentro do intervalo do negocio(6 meses)
e ele é descartado pois quando ele verifica se está dentro do intervalo ele retorna falso e o torna
não elegivel
e se a data de vencimento está dentro dos 6 meses fica mais facil pois se a data está dentro pode ser que ele não tenha pago
oque deixa mais preciso a classificação de bom ou mau.
"""
import time
import pandas as pd

print("1.Realizando imports")
start = time.time()

#

def retorna_values(dataset,coluna):
    values = dataset.iloc[:,coluna]
    return values

df = pd.read_csv("C:\CSCORE\TABELAO_CARTAO.txt", sep='\t', dtype='unicode')
print("2.DataFrames Carregados")

df['FLAG_ELEGIVEL'] = 0

#df1 = df[pd.to_datetime(df.DATA_PAGAMENTO) is pd.NaT ]
df1 = df[pd.to_datetime(df.DATA_VENCIMENTO) >= pd.to_datetime(df.DATA_CADASTRO)]
df1=  df1[pd.to_datetime(df1.DATA_VENCIMENTO) <= pd.to_datetime(df1.DATA_LIMITE)]

#df2 = df[pd.to_datetime(df.DATA_PAGAMENTO) >= pd.to_datetime(df.DATA_CADASTRO)]
#df2 = df2[pd.to_datetime(df2.DATA_PAGAMENTO) <= pd.to_datetime(df2.DATA_LIMITE)]

df = df1

progress = 1
for index, row in df.iterrows():
    if (pd.to_datetime(row['DATA_PAGAMENTO']) <= pd.to_datetime(row['DATA_LIMITE'])) & (pd.to_datetime(row['DATA_PAGAMENTO']) >= pd.to_datetime(row['DATA_CADASTRO'])):
        df.loc[index,'FLAG_ELEGIVEL'] = 0
    elif pd.to_datetime(row['DATA_PAGAMENTO']) is pd.NaT :
        if(pd.to_datetime(row['DATA_VENCIMENTO']) <= pd.to_datetime(row['DATA_LIMITE'])) & (pd.to_datetime(row['DATA_VENCIMENTO']) >= pd.to_datetime(row['DATA_CADASTRO'])):
            df.loc[index,'FLAG_ELEGIVEL'] = 0
        else:
            df.loc[index,'FLAG_ELEGIVEL'] = 1
    else:
        df.loc[index,'FLAG_ELEGIVEL'] = 1
    print((progress/739905)*100)
    progress = progress + 1
  
    

df.to_csv("C:\CSCORE\TABELAO_CARTAO_TARGET_ELEGIVEL.txt", sep='\t', encoding='utf-8')
print("3.Arquivo Criado") 

end = time.time()
print(end - start)
