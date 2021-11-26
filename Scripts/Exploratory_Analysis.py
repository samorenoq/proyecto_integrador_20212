#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.covariance import ledoit_wolf
from scipy.spatial.distance import mahalanobis

get_ipython().run_line_magic('matplotlib', 'inline')


# # Constantes

# Rutas de S3
S3_REFINED_URI = 's3://proyecto-integrador-20212-pregrado/datasets/refined/'


# # Carga de datos

# Cargar la matriz de precios y la matriz de retornos
df_prices = pd.read_parquet(f'{S3_REFINED_URI}matriz_precios.parquet')
df_returns = pd.read_parquet(f'{S3_REFINED_URI}matriz_retornos.parquet')


# # Fecha de corte para entrenamiento y validación
# Se quiere que el 80% de los datos se usen para el entrenamiento, para la selección de los portafolios y el 20% restante para la validación de dicha selección

# Separar los datos en entrenamiento y validación (no puede ser aleatorio por ser serie de tiempo)
train_size = 0.8
test_size = 1 - train_size
# Último índice de entrenamiento
last_train_index = int(df_returns.shape[0]*train_size)


# # Análisis exploratorio

# ## Gráficas

# Función para graficar cada acción
def plot_stock_matrix(df, save_filepath, num_rows=20, num_cols=5, figsize=(30,90)):
    # Número de acciones por gráfica
    stocks_per_plot = df.shape[1]//(num_rows*num_cols)+1
    # Subplots
    fig, ax = plt.subplots(num_rows, num_cols, figsize=figsize)
    # Contadores para la fila y la columna de las gráficas
    row, col = 0,0
    # Contador para el número de acciones por gráfica
    stock_count = 0
    for stock in df.columns:
        ax[row,col].plot(df[stock])
        stock_count += 1
        # Cambiar de gráfica si se llega al número de acciones por gráfica
        if stock_count == stocks_per_plot:
            stock_count = 0
            # Si la columna es la última, cambiar de fila
            if col == num_cols-1:
                col = 0
                row += 1
            else:
                col += 1
    
    plt.savefig(save_filepath)


plot_stock_matrix(df_prices, 'stock_prices_graphs.png')


plot_stock_matrix(df_returns, 'stock_returns_graphs.png')


# ## Análisis descriptivo

# Función para crear una descripción estadística de los datos
def describe_data(df):
    description = df.describe()
    description.loc['curtosis'] = stats.kurtosis(df)
    description.loc['asimetría'] = stats.skew(df)
    
    return description


# Descripción de los precios
describe_data(df_prices)


# Descripción de los retornos
describe_data(df_returns)


# ## Regresiones y correlación

# ### Matriz de correlación

corr_matrix = df_returns.corr()
corr_matrix


# Sacar la matriz triangular superior de la matriz de correlaciones para hacer análisis
upper_tri_corr = pd.DataFrame(np.triu(corr_matrix,1), index = df_returns.columns, columns=df_returns.columns)


# Hallar la acción con mayor y menor correlación a cada acción
max_corrs = upper_tri_corr.idxmax()[1:], upper_tri_corr.max()[1:]
min_corrs = upper_tri_corr.idxmin()[1:], upper_tri_corr.min()[1:]


max_min_corrs = pd.DataFrame()
max_min_corrs['Acción con máxima correlación'] = max_corrs[0]
max_min_corrs['Coeficiente de correlación máximo'] = max_corrs[1]
max_min_corrs['Acción con mínima correlación'] = min_corrs[0]
max_min_corrs['Coeficiente de correlación mínimo'] = min_corrs[1]


max_min_corrs.sort_values('Coeficiente de correlación máximo', ascending=False)


max_min_corrs.sort_values('Coeficiente de correlación mínimo')


# ### Regresión lineal de retornos para cada variable con las demás como explicativas

# Función para calcular el coeficiente de determinación para los retornos de una acción explicados por los de las demás
def calculate_r2(stock):
    # Matriz de variables independientes
    X = df_returns.drop(stock, axis = 1)
    # Vector con variable dependiente
    y = df_returns[stock]
    # Variables independientes de entrenamiento y validación
    X_train, X_test = X[:last_train_index], X[last_train_index:]
    # Variables dependientes de entrenamiento y validación
    y_train, y_test = y[:last_train_index], y[last_train_index:]
    # Modelo de regresión lineal
    lr = LinearRegression().fit(X_train, y_train)
    # Predicciones del modelo en entrenamiento
    train_prediction = lr.predict(X_train)
    # Predicciones del modelo en validación
    test_prediction = lr.predict(X_test)
    # Coeficiente de determinación en entrenamiento
    train_r2_score = r2_score(y_train, train_prediction)
    # Coeficiente de determinación en validación
    test_r2_score = r2_score(y_test, test_prediction)
    
    return train_r2_score, test_r2_score

# Función para calcular los coeficientes R2 de entrenamiento y validación para los retornos de todas las acciones
def calculate_r2_per_stock(df):
    stock_names = df.columns
    train_scores = np.zeros(df.shape[1])
    test_scores = np.zeros(df.shape[1])
    # Calcular los R2 para cada acción
    for i, stock in enumerate(stock_names):
        train_scores[i], test_scores[i] = calculate_r2(stock)
        
    ret = pd.DataFrame(index = stock_names)
    ret['R^2 entrenamiento'] = train_scores
    ret['R^2 validación'] = test_scores
    
    return ret


r2_per_stock = calculate_r2_per_stock(df_returns)


r2_per_stock


r2_per_stock.sort_values('R^2 validación', ascending=False)


# Guardar los resultados de los R2 en S3
r2_per_stock.to_parquet(f'{S3_REFINED_URI}coeficientes_r2_por_acción.parquet')


# ## Análisis de la matriz de covarianzas

# Calcular la matriz de varianzas y covarianzas de Ledoit & Wolf
cov_matrix = np.cov(df_returns.T)
# Calcular la matriz de varianzas y covarianzas de Ledoit & Wolf
cov_matrix = np.cov(df_returns.T)

# Guardar la matriz de varianzas y covarianzas de Ledoit & Wolf en S3
cov_matrix = pd.DataFrame(cov_matrix, index = df_returns.columns, columns = df_returns.columns)
cov_matrix.to_parquet(f'{S3_REFINED_URI}cov_retornos.parquet')

# Calcular la matriz de varianzas y covarianzas de Ledoit & Wolf
cov_matrix_lw = ledoit_wolf(df_returns)[0]

# Guardar la matriz de varianzas y covarianzas de Ledoit & Wolf en S3
cov_matrix_lw = pd.DataFrame(cov_matrix_lw, index = df_returns.columns, columns = df_returns.columns)
cov_matrix_lw.to_parquet(f'{S3_REFINED_URI}cov_retornos_lw.parquet')
# Guardar la matriz de varianzas y covarianzas de Ledoit & Wolf en S3
cov_matrix = pd.DataFrame(cov_matrix, index = df_returns.columns, columns = df_returns.columns)
cov_matrix.to_parquet(f'{S3_REFINED_URI}cov_retornos.parquet')


# Calcular la matriz de varianzas y covarianzas de Ledoit & Wolf
cov_matrix_lw = ledoit_wolf(df_returns)[0]

# Guardar la matriz de varianzas y covarianzas de Ledoit & Wolf en S3
cov_matrix_lw = pd.DataFrame(cov_matrix_lw, index = df_returns.columns, columns = df_returns.columns)
cov_matrix_lw.to_parquet(f'{S3_REFINED_URI}cov_retornos_lw.parquet')


print(f'El determinante de la matriz de covarianzas habitual es {np.linalg.det(cov_matrix)}')
print(f'El determinante de la matriz de covarianzas con shrinkage de Ledoit & Wolf es {np.linalg.det(cov_matrix_lw)}')
print(f'El número condición de la matriz de covarianzas habitual es {np.linalg.cond(cov_matrix)}')
print(f'El número condición de la matriz de covarianzas con shrinkage de Ledoit & Wolf es {np.linalg.cond(cov_matrix_lw)}')


# A pesar de que en este caso el determinante de la matriz no es bajo, el número condición es muy alto, lo cual quiere decir que la matriz está muy mal condicionada, lo cual implica que es muy sensible a cambios en los valores.

# ## Identificación de outliers
# 
# Se hará una identificación de valores atípicos con base en la distancia de Mahalanobis de cada vector al vector de medias.

# Función para encontrar outliers
def find_outliers(df, cov=cov_matrix_lw, distance='mahalanobis'): 
    # Calcular el vector de medias respecto a los días
    mean_vector = df.mean()
    # Sacar un vector de distancias entre cada día y el vector de medias
    distances_to_mean = np.zeros(df.shape[0])
    for i, day in enumerate(df.T):
        if distance=='mahalanobis':
            distances_to_mean[i] = mahalanobis(mean_vector, df.iloc[i], np.linalg.pinv(cov))
        else:
            distances_to_mean[i] = np.linalg.norm(mean_vector - df.iloc[i], ord=distance)
    
    return distances_to_mean


# Función para graficar outliers
def plot_outliers(percentile, distances_to_mean, df, distance_name = 'Mahalanobis'):
    # Valor de distancia por encima del cual se considera un outlier
    percentile_value = np.percentile(distances_to_mean, percentile)

    #Sacar los outliers muy lejos de la media
    outliers = np.where(distances_to_mean > percentile_value)
    #Sacar los datos que no son outliers
    normal_data = np.where(distances_to_mean <= percentile_value)

    fig, ax = plt.subplots(figsize=(10,10))

    ax.scatter(normal_data, distances_to_mean[normal_data], s=2);
    ax.scatter(outliers, distances_to_mean[outliers], color='r', s=2);

    ax.set_title(f'Distancia entre cada día y el día medio con {distance_name}')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Distancia');

    plt.savefig(f'Oultiers_{distance_name}.png')
    
    print(f'Los días más atípicos fueron: {df.iloc[outliers].index.format()}')


# Hallar las distancias de cada día al día medio con la distancia de Mahalanobis
distances_to_mean = find_outliers(df_returns)
# Hallar los outliers, siendo outliers los datos cuya distancia a la media está por encima de un percentil
plot_outliers(90, distances_to_mean, df_returns)


# Hallar las distancias de cada día al día medio con la distancia inducida por la norma-1
distances_to_mean = find_outliers(df_returns, distance=1)
# Hallar los outliers, siendo outliers los datos cuya distancia a la media está por encima de un percentil
plot_outliers(90, distances_to_mean, df_returns, 'Norma 1')


# Hallar las distancias de cada día al día medio con la distancia inducida por la norma-2
distances_to_mean = find_outliers(df_returns, distance=2)
# Hallar los outliers, siendo outliers los datos cuya distancia a la media está por encima de un percentil
plot_outliers(90, distances_to_mean, df_returns, 'Norma 2')


# Hallar las distancias de cada día al día medio con la distancia inducida por la norma-infinito
distances_to_mean = find_outliers(df_returns, distance=np.inf)
# Hallar los outliers, siendo outliers los datos cuya distancia a la media está por encima de un percentil
plot_outliers(90, distances_to_mean, df_returns, 'Norma Infinito')


extreme_outlier_index = np.argmax([np.linalg.norm(df_returns.mean()-df_returns.iloc[i], ord=2) for i in range(df_returns.shape[0])])


extreme_outlier_stock = df_returns.iloc[extreme_outlier_index].idxmax()
df_returns.iloc[extreme_outlier_index]


# Como se puede ver aquí, YNDX presenta problemas, entonces en el próximo filtrado de datos se eliminará. A continuación, se pueden ver los datos de outliers sacando YNDX

# Hallar las distancias de cada día al día medio con la distancia de Mahalanobis
distances_to_mean = find_outliers(df_returns.drop(extreme_outlier_stock, axis=1),
                                  cov=cov_matrix.drop(extreme_outlier_stock, axis=1).drop(extreme_outlier_stock))
# Hallar los outliers, siendo outliers los datos cuya distancia a la media está por encima de un percentil
plot_outliers(90, distances_to_mean, df_returns.drop(extreme_outlier_stock, axis=1), 'Mahalanobis (sin YNDX)')


# Hallar las distancias de cada día al día medio con la distancia inducida por la norma-1
distances_to_mean = find_outliers(df_returns.drop(extreme_outlier_stock, axis=1),
                                  cov=cov_matrix.drop(extreme_outlier_stock, axis=1).drop(extreme_outlier_stock),
                                  distance=1)
# Hallar los outliers, siendo outliers los datos cuya distancia a la media está por encima de un percentil
plot_outliers(90, distances_to_mean, df_returns, 'Norma 1 (sin YNDX)')


# Hallar las distancias de cada día al día medio con la distancia inducida por la norma-2
distances_to_mean = find_outliers(df_returns.drop(extreme_outlier_stock, axis=1),
                                  cov=cov_matrix.drop(extreme_outlier_stock, axis=1).drop(extreme_outlier_stock),
                                  distance=2)
# Hallar los outliers, siendo outliers los datos cuya distancia a la media está por encima de un percentil
plot_outliers(90, distances_to_mean, df_returns, 'Norma 2 (sin YNDX)')


# Hallar las distancias de cada día al día medio con la distancia inducida por la norma-infinito
distances_to_mean = find_outliers(df_returns.drop(extreme_outlier_stock, axis=1),
                                  cov=cov_matrix.drop(extreme_outlier_stock, axis=1).drop(extreme_outlier_stock),
                                  distance=np.inf)
# Hallar los outliers, siendo outliers los datos cuya distancia a la media está por encima de un percentil
plot_outliers(90, distances_to_mean, df_returns, 'Norma Infinito (sin YNDX)')


# # Conclusiones del análisis

# * Activos a eliminar:
#     - VTI: es un índice compuesto por muchas acciones
#     - SPY: es un ETF que sigue al S&P500, entonces contiene muchas de las acciones que ya están por otro lado
#     - DIA: es un ETF que sigue al índice Dow Jones, entonces contiene muchas de las acciones que ya están por otro lado
#     - IWN: es un ETF que sigue al índice Russel 2000 Value, entonces contiene muchas de las acciones que ya están por otro lado
#     - IJH: es un ETF que sigue al índice S&P MidCap 400, entonces contiene muchas de las acciones que ya están por otro lado
#     - IWF: es un ETF que sigue al índice Russel 1000 Growth, entonces contiene muchas de las acciones que ya están por otro lado
#     - GOOG: es la acción de Google clase C (la clase A, GOOGL, también está, pero no se eliminará)
#     - YNDX: tiene un valor muy atípico que debe ser un error

# Activos a eliminar
to_remove = ['VTI', 'SPY', 'DIA', 'IWN', 'IJH', 'IWF', 'GOOG', 'YNDX']
# Eliminar las acciones definidas anteriormente y crear una nueva matriz de precios y de retornos y matrices de covarianza
df_prices_filtered_train = df_prices.drop(to_remove, axis=1).iloc[:last_train_index]
df_returns_filtered_train = df_returns.drop(to_remove, axis=1).iloc[:last_train_index]

df_prices_filtered_train.to_parquet(f'{S3_REFINED_URI}matriz_precios_filtrada_train.parquet')
df_returns_filtered_train.to_parquet(f'{S3_REFINED_URI}matriz_retornos_filtrada_train.parquet')

df_prices_filtered_test = df_prices.drop(to_remove, axis=1).iloc[last_train_index:]
df_returns_filtered_test = df_returns.drop(to_remove, axis=1).iloc[last_train_index:]

df_prices_filtered_test.to_parquet(f'{S3_REFINED_URI}matriz_precios_filtrada_test.parquet')
df_returns_filtered_test.to_parquet(f'{S3_REFINED_URI}matriz_retornos_filtrada_test.parquet')


# Calcular la matriz de varianzas y covarianzas con datos de entrenamiento
cov_matrix_filtered = np.cov(df_returns_filtered_train.T)

# Guardar la matriz de varianzas y covarianzas de con datos de entrenamiento en S3
cov_matrix_filtered = pd.DataFrame(cov_matrix_filtered,
                                   index = df_returns_filtered_train.columns, columns = df_returns_filtered_train.columns)
cov_matrix_filtered.to_parquet(f'{S3_REFINED_URI}cov_retornos_filtrada.parquet')


# Calcular la matriz de varianzas y covarianzas de Ledoit & Wolf
cov_matrix_lw_filtered = ledoit_wolf(df_returns_filtered_train)[0]

# Guardar la matriz de varianzas y covarianzas de Ledoit & Wolf en S3
cov_matrix_lw_filtered = pd.DataFrame(cov_matrix_lw_filtered,
                                      index = df_returns_filtered_train.columns, columns = df_returns_filtered_train.columns)
cov_matrix_lw_filtered.to_parquet(f'{S3_REFINED_URI}cov_retornos_lw_filtrada.parquet')

