#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# # Constantes

# Rutas de S3
S3_REFINED_URI = 's3://proyecto-integrador-20212-pregrado/datasets/refined/'


# # Clusterización supervisada de las acciones con KNN
# En este caso se usará cada día como una variable y cada acción como un registro porque se quieren clasificar las acciones en diferentes clusters.

# Cargar la matriz de retornos etiquetada de S3
df_returns_with_tags = pd.read_parquet(f'{S3_REFINED_URI}matriz_retornos_etiquetada.parquet')
df_returns_with_tags.head()


# Definir X y Y
X = df_returns_with_tags.drop('label', axis = 1)
y = df_returns_with_tags['label']


# Separar los datos en datos de entrenamiento y de validación, con una separación de 80% para entrenamiento y 20% para validación

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)


# Función calcular la precisión en entrenamiento y validación para un modelo dado
def calculate_accuracy(model, X_train, X_test, y_train, y_test):
    return model.score(X_train, y_train), model.score(X_test, y_test)


# Hallar la precisión del modelo KNN para varios valores de k
num_ks = 30
ks = np.arange(1, num_ks+1)
train_accuracies = np.zeros(num_ks)
test_accuracies = np.zeros(num_ks)

for i, k in enumerate(ks):
    knn = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    train_accuracies[i], test_accuracies[i] = calculate_accuracy(knn, X_train, X_test, y_train, y_test)


fig, ax = plt.subplots(figsize=(12,7))
ax.set_title('Precisión de clasificador KNN para cada $k$')
ax.set_xlabel('k')
ax.set_xticks(np.linspace(1,num_ks, num_ks))
ax.set_ylabel('Precisión (%)')
ax.set_ylim(0,100)
ax.plot(ks, train_accuracies*100, label='Entrenamiento')
ax.plot(ks, test_accuracies*100, label='Validación')
ax.legend()
plt.savefig('KNN_Accuracy.png')

