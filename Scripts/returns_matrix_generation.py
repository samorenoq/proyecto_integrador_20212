#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import os


SAMPLE_SIZE = 5
PORTFOLIO_SIZE = 5
STOCKS_FILEPATH = 'daily_data/'
S3_TRUSTED_URI = 's3://proyecto-integrador-20212-pregrado/datasets/trusted/'
S3_REFINED_URI = 's3://proyecto-integrador-20212-pregrado/datasets/refined/'


# Lista de los nombres de cada archivo
stocks_list = os.listdir(STOCKS_FILEPATH)
stock_names = np.array([stock.split('-')[0] for stock in stocks_list])


# Cargar la matriz de precios
df_prices = pd.read_parquet(S3_REFINED_URI+'matriz_precios.parquet')
df_prices


# Crear la matriz de retornos diarios
df_returns = df_prices.pct_change(1).iloc[1:]
df_returns


# Escribir la matriz de retornos en S3
df_returns.to_parquet(S3_REFINED_URI+'matriz_retornos.parquet')

