#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.covariance import ledoit_wolf


# Número de acciones en el portafolio
PORTFOLIO_SIZE = 300
# Número de portafolios a generar
NUM_PORTFOLIOS = 50
# Rutas de S3
S3_REFINED_URI = 's3://proyecto-integrador-20212-pregrado/datasets/refined/'


# # Entrenamiento

# Cargar la matriz de precios
df_returns_train = pd.read_parquet(S3_REFINED_URI+'matriz_retornos_filtrada_train.parquet')
df_returns_train


# Crear un arreglo con los nombres de las acciones
stock_names = np.array(df_returns_train.columns)


# Función para sacar un portafolio de PORTFOLIO_SIZE acciones escogidas aleatoriamente
def select_random_stocks(stock_names, n_stocks=PORTFOLIO_SIZE):
    return np.random.choice(stock_names, size=PORTFOLIO_SIZE, replace=False)


# Generación de <NUM_PORTFOLIOS> porfafolios de <PORTFOLIO_SIZE> acciones
portfolios = [df_returns_train[select_random_stocks(stock_names)] for i in range(NUM_PORTFOLIOS)]


# Calcular la matriz de covarianza para cada portafolio, con el método habitual y con shrinkage de Ledoit & Wolf
cov_matrices = [i for i in range(NUM_PORTFOLIOS)]
cov_matrices_lw = [i for i in range(NUM_PORTFOLIOS)]

for i, portfolio in enumerate(portfolios):
    cov_matrices[i] = pd.DataFrame(np.cov(portfolio.T), index=portfolio.columns, columns=portfolio.columns)
    cov_matrices_lw[i] = pd.DataFrame(ledoit_wolf(portfolio)[0], index=portfolio.columns, columns=portfolio.columns)


# Guardar cada portafolio de entrenamiento y sus matrices de covarianzas en la zona Refined
for i, portfolio in enumerate(portfolios):
    print(i, end=', ')
    portfolio.to_parquet(f'{S3_REFINED_URI}portfolio_{i}_returns_train.parquet')
    cov_matrices[i].to_parquet(f'{S3_REFINED_URI}portfolio_{i}_cov.parquet')
    cov_matrices_lw[i].to_parquet(f'{S3_REFINED_URI}portfolio_{i}_cov_lw.parquet')


# # Validación

# Cargar la matriz de precios
df_returns_test = pd.read_parquet(S3_REFINED_URI+'matriz_retornos_filtrada_test.parquet')
df_returns_test


# Crear un arreglo con los nombres de las acciones
stock_names = np.array(df_returns_test.columns)


# Función para sacar un portafolio de PORTFOLIO_SIZE acciones escogidas aleatoriamente
def select_random_stocks(stock_names, n_stocks=PORTFOLIO_SIZE):
    return np.random.choice(stock_names, size=PORTFOLIO_SIZE, replace=False)


# Generación de <NUM_PORTFOLIOS> porfafolios de <PORTFOLIO_SIZE> acciones
portfolios = [df_returns_test[select_random_stocks(stock_names)] for i in range(NUM_PORTFOLIOS)]


# Guardar cada portafolio de entrenamiento y sus matrices de covarianzas en la zona Refined
for i, portfolio in enumerate(portfolios):
    print(i, end=', ')
    portfolio.to_parquet(f'{S3_REFINED_URI}portfolio_{i}_returns_test.parquet')

