#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import os
import datetime


STOCKS_FILEPATH = '../../daily_data/'
S3_TRUSTED_URI = 's3://proyecto-integrador-20212-pregrado/datasets/trusted/'
S3_REFINED_URI = 's3://proyecto-integrador-20212-pregrado/datasets/refined/'


# Lista de los nombres de cada archivo
stocks_list = os.listdir(STOCKS_FILEPATH)
stock_names = np.array([stock.split('-')[0] for stock in stocks_list])


# Crear dataframe con todas las acciones
df = pd.DataFrame()

for stock in stocks_list:
    stock_name = stock.split('-')[0]
    df[stock_name] = pd.read_parquet(os.path.join(STOCKS_FILEPATH, stock))['close']
    #df[stock_name] = pd.read_parquet(S3_TRUSTED_URI+stock)['close']


# Función para hallar el número de valores NA para cada acción
def na_count(df):
    return np.array([len(df[stock_name][df[stock_name].isna()]) for stock_name in stock_names])


# Fecha inicial a partir de la cual se tomarán las acciones
# (teniendo en cuenta que muchas no existían en años anteriores)
initial_date = datetime.date(2014,1,1)
# Filtrar las fechas para incluir fechas posteriores a la fecha inicial
df_after_date = df[df.index > initial_date]
df_after_date.shape


# Hallar el número de días con NA para cada acción
nas = na_count(df_after_date)
# Hallar las acciones que más días faltantes tienen
nas_indices = np.argsort(nas)[::-1] 

nas_indices


# Hallar el número de acciones a las que les faltan muchos días de datos
stock_indices_to_remove = nas_indices[nas[nas_indices] > 63]
print(f'Se eliminarán {len(stock_indices_to_remove)} acciones de la lista de acciones elegibles')


# Sacar de la lista las acciones a las que les falten muchos días de datos
stocks_to_remove = stock_names[stock_indices_to_remove]
df_applicable_stocks = df_after_date.drop(stocks_to_remove, axis=1)
df_applicable_stocks.shape


# Llenar los valores NA de las acciones restantes con el precio anterior / siguiente
df_applicable_stocks_no_na = df_applicable_stocks.fillna(method='ffill').fillna(method='bfill')


# Guardar la matriz de precios como un CSV
df_applicable_stocks_no_na.to_parquet(S3_REFINED_URI+'matriz_precios.parquet')

