import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn import linear_model
from sklearn.metrics import r2_score
import seaborn as sns
import pandas as pd
import xlrd
import statistics as stt
import time
import datetime


def retorna_values(dataset,coluna):
    values = dataset.iloc[:,coluna]
    return values

dataCadastral = pd.read_csv('CADASTRAL.txt', sep='\t', dtype='unicode')

dataClienteDataCad = retorna_values(dataCadastral,[0,9])

df = dataClienteDataCad.head(n=300000)


lista_index = []
for index, row in df.iterrows():
    print("===Index===")
    print(index)
    row["DATA_CADASTRO"] = pd.to_datetime(row["DATA_CADASTRO"])
    dataCadastro = pd.to_datetime(row["DATA_CADASTRO"])
    dataLimiteInferior = pd.to_datetime("2017/01/01")
    print("==== Data Cadastro ===")
    print(dataCadastro)
    if dataCadastro < dataLimiteInferior:
        lista_index.append(index)
        print("Se data antes 2017: Drop")
    elif dataCadastro is pd.NaT:
        lista_index.append(index)
        print("Se não, se NAT: Drop")
    else:
        print("Se não, Mantem")
    print("=======")

df.drop(lista_index).to_csv("CLIENTES_2017", sep='\t', encoding='utf-8')

end = time.time()
print(end - start)
