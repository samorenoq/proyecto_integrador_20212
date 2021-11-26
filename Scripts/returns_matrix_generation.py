#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import os


SAMPLE_SIZE = 5
PORTFOLIO_SIZE = 5
S3_TRUSTED_URI = 's3://proyecto-integrador-20212-pregrado/datasets/trusted/'
S3_REFINED_URI = 's3://proyecto-integrador-20212-pregrado/datasets/refined/'


# Cargar la matriz de precios
df_prices = pd.read_parquet(S3_REFINED_URI+'matriz_precios.parquet')
df_prices


# Nombres de las acciones
stock_names = np.array(df_prices.columns)


# Crear la matriz de retornos diarios
frequency = 1
df_returns = df_prices.pct_change(frequency).iloc[frequency:]
df_returns


# Escribir la matriz de retornos diarios en S3
df_returns.to_parquet(S3_REFINED_URI+'matriz_retornos.parquet')

