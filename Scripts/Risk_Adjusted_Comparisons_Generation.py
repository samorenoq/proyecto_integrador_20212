#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.covariance import ledoit_wolf


# # Constantes

# Número de acciones en el portafolio
PORTFOLIO_SIZE = 300
# Número de portafolios a generar
NUM_PORTFOLIOS = 50
# Rutas de S3
S3_REFINED_URI = 's3://proyecto-integrador-20212-pregrado/datasets/refined/'
# Número de días usados para calcular la rentabilidad y desviación estándar
LOOKBACK_PERIOD = 252


# # Cargar portafolios y matrices de covarianzas de S3

# Cargar los portafolios aleatorios de entrenamiento
portfolios_train = [i for i in range(NUM_PORTFOLIOS)]
for i in portfolios_train:
    print(i, end=', ')
    portfolios_train[i] = pd.read_parquet(S3_REFINED_URI+f'portfolio_{i}_returns_train.parquet')


# Cargar los portafolios aleatorios de validación
portfolios_test = [i for i in range(NUM_PORTFOLIOS)]
for i in portfolios_test:
    print(i, end=', ')
    portfolios_test[i] = pd.read_parquet(S3_REFINED_URI+f'portfolio_{i}_returns_test.parquet')


# Cargar las matrices de varianzas y covarianzas de S3
cov_matrices = [i for i in range(NUM_PORTFOLIOS)]
cov_matrices_lw = [i for i in range(NUM_PORTFOLIOS)]

for i in cov_matrices:
    print(i, end=', ')
    cov_matrices[i] = pd.read_parquet(S3_REFINED_URI+f'portfolio_{i}_cov.parquet')
    cov_matrices_lw[i] = pd.read_parquet(S3_REFINED_URI+f'portfolio_{i}_cov_lw.parquet')


# # Análisis de las matrices de varianzas y covarianzas

# Función para calcular el determinante y el número condición de la matriz de varianzas y covarianzas de cada portafolio
def calculate_determinants_and_condition_nums(cov_matrices, cov_matrices_lw, num_portfolios=NUM_PORTFOLIOS):
    dets = np.zeros(num_portfolios)
    conds = np.zeros(num_portfolios)
    dets_lw = np.zeros(num_portfolios)
    conds_lw = np.zeros(num_portfolios)

    for i, cov in enumerate(cov_matrices):
        dets[i] = np.linalg.det(cov)
        conds[i] = np.linalg.cond(cov)

    for i, cov_lw in enumerate(cov_matrices_lw):
        dets_lw[i] = np.linalg.det(cov_lw)
        conds_lw[i] = np.linalg.cond(cov_lw)
        
    ret = pd.DataFrame(index = [f'portfolio_{i}' for i in range(num_portfolios)])
    ret['Determinante cov habitual'] = dets
    ret['Determinante cov LW'] = dets_lw
    ret['Número condición cov habitual'] = conds
    ret['Número condición cov LW'] = conds_lw
    
    return ret


determinants_condition_numbers = calculate_determinants_and_condition_nums(cov_matrices, cov_matrices_lw)
determinants_condition_numbers


# Escribir la matriz de determinantes y números condición a S3
determinants_condition_numbers.to_parquet(f'{S3_REFINED_URI}matriz_determinantes_números_condición.parquet')


# # Funciones

# Función para sacar un portafolio de PORTFOLIO_SIZE acciones escogidas aleatoriamente
def select_random_stocks(stock_names, n_stocks=PORTFOLIO_SIZE):
    return np.random.choice(stock_names, size=PORTFOLIO_SIZE, replace=False)

# Función para calcular el retorno promedio del portafolio durante los últimos <window> períodos
def portfolio_mean_returns(df_returns, ws, window=LOOKBACK_PERIOD):
    return pd.Series(np.dot(df_returns, ws), index=df_returns.index).rolling(window).mean().iloc[window:]


# Función para calcular la desviación estándar del portafolio
def portfolio_std(df_returns, ws, cov, window=LOOKBACK_PERIOD):
    # Retornar la desviación estándar diaria del portafolio
    return (ws.T.dot(cov).dot(ws)/LOOKBACK_PERIOD)**.5

# Función para calcular los retornos promedio para una lista de portafolios
def calculate_portfolio_returns(portfolios, ws):
    portfolio_returns_matrix = pd.DataFrame()

    for i, portfolio in enumerate(portfolios):
        portfolio_name = f'portfolio_{i}'
        portfolio_returns_matrix[portfolio_name] = portfolio_mean_returns(portfolio, ws[i])
        
    return portfolio_returns_matrix

# Función para calcular las desviaciones estándar de una lista de portafolios
def calculate_portfolio_std(portfolios, ws, cov_matrices):
    portfolio_stds_matrix = pd.Series(index=[f'portfolio_{i}' for i in range(NUM_PORTFOLIOS)],
                                            dtype=np.float64)

    for i, portfolio in enumerate(portfolios):
        portfolio_stds_matrix[i] = portfolio_std(portfolio, ws[i], cov_matrices[i])
        
    return portfolio_stds_matrix

# Función para calcular los retornos diarios promedio ajustados por riesgo
def calculate_portfolio_risk_adjusted_returns(portfolio_returns_matrix, portfolio_stds_matrix):
    return portfolio_returns_matrix/portfolio_stds_matrix

# Función para calcular los pesos del portafolio de mínima varianza para una matriz de retornos
def calculate_minimum_variance_weights(portfolios, cov_matrices, moore_penrose=False):
    ws_minimum_variance = [0 for i in portfolios]
    ones = np.ones(PORTFOLIO_SIZE)
    for i, portfolio in enumerate(portfolios):
        if moore_penrose:
            cov_inv = np.linalg.pinv(cov_matrices[i])
        else:
            cov_inv = np.linalg.inv(cov_matrices[i])
            
        numerator = ones @ cov_inv
        denominator = ones.T @ cov_inv @ ones
        
        ws_minimum_variance[i] = numerator/denominator
        
    return ws_minimum_variance


# # Resultados en entrenamiento

# ## Portafolio con pesos iguales

# Calcular el peso de cada activo para el escenario de igualdad de pesos
ws_constant = np.array([np.ones(PORTFOLIO_SIZE)*(1/PORTFOLIO_SIZE) for i in portfolios_train])
print(f'Cada acción tendrá un peso de {(ws_constant[0][0]*100).round(2)}%')


# Calcular el retorno promedio de los últimos <LOOKBACK_PERIOD> días para cada fecha para cada portafolio
portfolio_returns_equal_weights = calculate_portfolio_returns(portfolios_train, ws_constant)


portfolio_returns_equal_weights


# Calcular las desviaciones estándar de cada portafolio
portfolio_stds_equal_weights = calculate_portfolio_std(portfolios_train, ws_constant, cov_matrices)


# Calcular los retornos diarios promedio ajustados por riesgo
portfolio_risk_adjusted_returns_equal_weights =     calculate_portfolio_risk_adjusted_returns(portfolio_returns_equal_weights, portfolio_stds_equal_weights)


portfolio_risk_adjusted_returns_equal_weights


# Guardar la matriz de retornos ajustados por riesgo para cada portafolio en S3
portfolio_risk_adjusted_returns_equal_weights.to_parquet(
    S3_REFINED_URI+'matriz_retornos_ajustados_por_riesgo_pesos_iguales_train.parquet')


# ## Portafolio de mínima varianza tradicional

# El vector de pesos $w$ del portafolio de mínima varianza está dado por
# $$
# w_{MV} = \frac{\Sigma^{-1}1}{1^T\Sigma^{-1}1}
# $$
# donde,<br>
# $\Sigma^{-1}$ es la matriz inversa de la matriz de covarianzas de orden $n \times n$<br>
# $1$ es un vector de 1s de orden $n \times 1$

# Calcular los pesos para el portafolio de mínima varianza
ws_minimum_variance = calculate_minimum_variance_weights(portfolios_train, cov_matrices)


# Calcular el retorno promedio de los últimos <LOOKBACK_PERIOD> días para cada fecha para cada portafolio
portfolio_returns_minimum_variance = calculate_portfolio_returns(portfolios_train, ws_minimum_variance)


portfolio_returns_minimum_variance


# Calcular las desviaciones estándar de cada portafolio
portfolio_stds_minimum_variance = calculate_portfolio_std(portfolios_train, ws_minimum_variance, cov_matrices)


# Calcular los retornos diarios promedio ajustados por riesgo
portfolio_risk_adjusted_returns_minimum_variance =     calculate_portfolio_risk_adjusted_returns(portfolio_returns_minimum_variance, portfolio_stds_minimum_variance)


portfolio_risk_adjusted_returns_minimum_variance


# Guardar la matriz de retornos ajustados por riesgo para cada portafolio en S3
portfolio_risk_adjusted_returns_equal_weights.to_parquet(
    S3_REFINED_URI+'matriz_retornos_ajustados_por_riesgo_varianza_mínima_train.parquet')


# ## Portafolio con shrinkage de Ledoit & Wolf

# Calcular los pesos para el portafolio de mínima varianza con Ledoit & Wolf
ws_minimum_variance_lw = calculate_minimum_variance_weights(portfolios_train, cov_matrices_lw, moore_penrose=True)


# Calcular el retorno promedio de los últimos <LOOKBACK_PERIOD> días para cada fecha para cada portafolio
portfolio_returns_minimum_variance_lw = calculate_portfolio_returns(portfolios_train, ws_minimum_variance_lw)


portfolio_returns_minimum_variance


# Calcular las desviaciones estándar de cada portafolio
portfolio_stds_minimum_variance_lw = calculate_portfolio_std(portfolios_train, ws_minimum_variance_lw, cov_matrices)


# Calcular los retornos diarios promedio ajustados por riesgo
portfolio_risk_adjusted_returns_minimum_variance_lw =     calculate_portfolio_risk_adjusted_returns(portfolio_returns_minimum_variance_lw, portfolio_stds_minimum_variance_lw)


portfolio_risk_adjusted_returns_minimum_variance_lw


# Guardar la matriz de retornos ajustados por riesgo para cada portafolio en S3
portfolio_risk_adjusted_returns_equal_weights.to_parquet(
    S3_REFINED_URI+'matriz_retornos_ajustados_por_riesgo_varianza_mínima_lw_train.parquet')


# Retornos ajustados por riesgo promedio con Ledoit & Wolf vs mínima varianza
risk_adjusted_lw_minus_min_var =     portfolio_risk_adjusted_returns_minimum_variance_lw.mean()-portfolio_risk_adjusted_returns_minimum_variance.mean()


# Retornos ajustados por riesgo promedio con mínima varianza vs pesos iguales
risk_adjusted_min_var_minus_equal_weights =     portfolio_risk_adjusted_returns_minimum_variance.mean()-portfolio_risk_adjusted_returns_equal_weights.mean()


# Retornos ajustados por riesgo promedio con Ledoit & Wolf vs pesos iguales
risk_adjusted_lw_minus_equal_weights =     portfolio_risk_adjusted_returns_minimum_variance_lw.mean()-portfolio_risk_adjusted_returns_equal_weights.mean()


# Juntar los resultados en un DataFrame
risk_adjusted_comparisons = pd.DataFrame()
risk_adjusted_comparisons['Mínima varianza - Pesos iguales'] = risk_adjusted_min_var_minus_equal_weights
risk_adjusted_comparisons['Mínima varianza con LW - Pesos iguales'] = risk_adjusted_lw_minus_equal_weights
risk_adjusted_comparisons['Mínima varianza con LW - Mínima Varianza'] = risk_adjusted_lw_minus_min_var
risk_adjusted_comparisons.loc[len(risk_adjusted_comparisons)] = risk_adjusted_comparisons.mean()

risk_adjusted_comparisons.index = list(risk_adjusted_comparisons.index[:-1])+['Promedio']

risk_adjusted_comparisons


# Escribir comparaciones a S3
risk_adjusted_comparisons.to_parquet(f'{S3_REFINED_URI}comparaciones_retorno_ajustado_por_riesgo_train.parquet')


# # Resultados en validación

# ## Portafolio con pesos iguales

# Calcular el peso de cada activo para el escenario de igualdad de pesos
ws_constant = np.array([np.ones(PORTFOLIO_SIZE)*(1/PORTFOLIO_SIZE) for i in portfolios_test])
print(f'Cada acción tendrá un peso de {(ws_constant[0][0]*100).round(2)}%')


# Calcular el retorno promedio de los últimos <LOOKBACK_PERIOD> días para cada fecha para cada portafolio
portfolio_returns_equal_weights = calculate_portfolio_returns(portfolios_test, ws_constant)


portfolio_returns_equal_weights


# Calcular las desviaciones estándar de cada portafolio
portfolio_stds_equal_weights = calculate_portfolio_std(portfolios_test, ws_constant, cov_matrices)


# Calcular los retornos diarios promedio ajustados por riesgo
portfolio_risk_adjusted_returns_equal_weights =     calculate_portfolio_risk_adjusted_returns(portfolio_returns_equal_weights, portfolio_stds_equal_weights)


portfolio_risk_adjusted_returns_equal_weights


# Guardar la matriz de retornos ajustados por riesgo para cada portafolio en S3
portfolio_risk_adjusted_returns_equal_weights.to_parquet(
    S3_REFINED_URI+'matriz_retornos_ajustados_por_riesgo_pesos_iguales_test.parquet')


# ## Portafolio de mínima varianza tradicional

# El vector de pesos $w$ del portafolio de mínima varianza está dado por
# $$
# w_{MV} = \frac{\Sigma^{-1}1}{1^T\Sigma^{-1}1}
# $$
# donde,<br>
# $\Sigma^{-1}$ es la matriz inversa de la matriz de covarianzas de orden $n \times n$<br>
# $1$ es un vector de 1s de orden $n \times 1$

# Calcular los pesos para el portafolio de mínima varianza
ws_minimum_variance = calculate_minimum_variance_weights(portfolios_test, cov_matrices)


# Calcular el retorno promedio de los últimos <LOOKBACK_PERIOD> días para cada fecha para cada portafolio
portfolio_returns_minimum_variance = calculate_portfolio_returns(portfolios_test, ws_minimum_variance)


portfolio_returns_minimum_variance


# Calcular las desviaciones estándar de cada portafolio
portfolio_stds_minimum_variance = calculate_portfolio_std(portfolios_test, ws_minimum_variance, cov_matrices)


# Calcular los retornos diarios promedio ajustados por riesgo
portfolio_risk_adjusted_returns_minimum_variance =     calculate_portfolio_risk_adjusted_returns(portfolio_returns_minimum_variance, portfolio_stds_minimum_variance)


portfolio_risk_adjusted_returns_minimum_variance


# Guardar la matriz de retornos ajustados por riesgo para cada portafolio en S3
portfolio_risk_adjusted_returns_equal_weights.to_parquet(
    S3_REFINED_URI+'matriz_retornos_ajustados_por_riesgo_varianza_mínima_test.parquet')


# ## Portafolio con shrinkage de Ledoit & Wolf

# Calcular los pesos para el portafolio de mínima varianza con Ledoit & Wolf
ws_minimum_variance_lw = calculate_minimum_variance_weights(portfolios_test, cov_matrices_lw, moore_penrose=True)


# Calcular el retorno promedio de los últimos <LOOKBACK_PERIOD> días para cada fecha para cada portafolio
portfolio_returns_minimum_variance_lw = calculate_portfolio_returns(portfolios_test, ws_minimum_variance_lw)


portfolio_returns_minimum_variance


# Calcular las desviaciones estándar de cada portafolio
portfolio_stds_minimum_variance_lw = calculate_portfolio_std(portfolios_test, ws_minimum_variance_lw, cov_matrices)


# Calcular los retornos diarios promedio ajustados por riesgo
portfolio_risk_adjusted_returns_minimum_variance_lw =     calculate_portfolio_risk_adjusted_returns(portfolio_returns_minimum_variance_lw, portfolio_stds_minimum_variance_lw)


portfolio_risk_adjusted_returns_minimum_variance_lw


# Guardar la matriz de retornos ajustados por riesgo para cada portafolio en S3
portfolio_risk_adjusted_returns_equal_weights.to_parquet(
    S3_REFINED_URI+'matriz_retornos_ajustados_por_riesgo_varianza_mínima_lw_test.parquet')


# Retornos ajustados por riesgo promedio con Ledoit & Wolf vs mínima varianza
risk_adjusted_lw_minus_min_var =     portfolio_risk_adjusted_returns_minimum_variance_lw.mean()-portfolio_risk_adjusted_returns_minimum_variance.mean()


# Retornos ajustados por riesgo promedio con mínima varianza vs pesos iguales
risk_adjusted_min_var_minus_equal_weights =     portfolio_risk_adjusted_returns_minimum_variance.mean()-portfolio_risk_adjusted_returns_equal_weights.mean()


# Retornos ajustados por riesgo promedio con Ledoit & Wolf vs pesos iguales
risk_adjusted_lw_minus_equal_weights =     portfolio_risk_adjusted_returns_minimum_variance_lw.mean()-portfolio_risk_adjusted_returns_equal_weights.mean()


# Juntar los resultados en un DataFrame
risk_adjusted_comparisons = pd.DataFrame()
risk_adjusted_comparisons['Mínima varianza - Pesos iguales'] = risk_adjusted_min_var_minus_equal_weights
risk_adjusted_comparisons['Mínima varianza con LW - Pesos iguales'] = risk_adjusted_lw_minus_equal_weights
risk_adjusted_comparisons['Mínima varianza con LW - Mínima Varianza'] = risk_adjusted_lw_minus_min_var
risk_adjusted_comparisons.loc[len(risk_adjusted_comparisons)] = risk_adjusted_comparisons.mean()

risk_adjusted_comparisons.index = list(risk_adjusted_comparisons.index[:-1])+['Promedio']

risk_adjusted_comparisons


# Escribir comparaciones a S3
risk_adjusted_comparisons.to_parquet(f'{S3_REFINED_URI}comparaciones_retorno_ajustado_por_riesgo_test.parquet')

