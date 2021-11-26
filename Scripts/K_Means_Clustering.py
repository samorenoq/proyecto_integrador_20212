#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import datetime

get_ipython().run_line_magic('matplotlib', 'inline')


# # Constantes

# Rutas de S3
S3_REFINED_URI = 's3://proyecto-integrador-20212-pregrado/datasets/refined/'


# # Clusterización de las acciones con K-means
#  En este caso se usará cada día como una variable y cada acción como un registro porque se quieren clasificar las acciones en diferentes clusters.

# Cargar los retornos
df_returns = pd.read_parquet(f'{S3_REFINED_URI}matriz_retornos_filtrada.parquet')
df_returns.head()


# Como cada acción es una observación, se debe usar la transpuesta
data = df_returns.T


# Definición de k
k = 11
# Clustering con K-Means
labels = KMeans(n_clusters=k, random_state=0).fit_predict(data)
# Agregar las etiquetas como columna a los datos
data['label'] = labels


# Graficar los resultados usando como coordenadas los dos componentes principales
pca = PCA(2)
pca_data = pca.fit_transform(data)


fig, ax = plt.subplots(figsize=(10,10))

for lab in np.unique(labels):
    ax.scatter(pca_data[labels == lab, 0],pca_data[labels == lab, 1], label=lab)

ax.set_title('Clustering de Acciones con K-Means ($k = 11$)')
ax.legend()
plt.savefig('K_Means_Clustering.png')
plt.show()


# A pesar de que en algunos casos hay clusters claros, en otros no hay una división tan clara de las acciones y los clusters a los que pertenecen. Esto también puede pasar porque la clusterización fue hecha en una dimensión mucho más alta que la que se puede graficar.

# Convertir las columnas en texto
string_dates = list(map(lambda x: x.strftime('%Y-%m-%d'), df_returns.index)) + ['label']
data.columns = string_dates

# Guardar la matriz de retornos con etiquetas en S3
data.to_parquet(f'{S3_REFINED_URI}matriz_retornos_etiquetada.parquet')

