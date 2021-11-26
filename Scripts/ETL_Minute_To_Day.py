#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import os
import boto3


# # Constantes

STOCKS_FILEPATH = '../../2020-11-25/'
S3_RAW_PATH = 's3://proyecto-integrador-20212-pregrado/datasets/raw/'
S3_TRUSTED_PATH = 's3://proyecto-integrador-20212-pregrado/datasets/trusted/'


stocks_list = os.listdir(STOCKS_FILEPATH)


# # Funciones

# Función para obtener las rutas de todos los archivos de un directorio en S3
def get_s3_paths(stocks_list):
    # Lista de rutas fuente en S3
    sources = [S3_RAW_PATH+stock for stock in stocks_list]
    # Lista de rutas de destino en S3
    destinations = [S3_TRUSTED_PATH+stock for stock in stocks_list]
    #Lista de tuplas (fuente, destino)
    return [(source, destination) for source, destination in zip(sources, destinations)]

# Función para pasar los datos de minutos a datos diarios, de raw a trusted
def minutes_to_days(path_tuple):
    source, destination = path_tuple
    # Leer datos de AWS
    df = pd.read_parquet(source)
    # Crear una columna con las fechas (sin hora)
    df['date'] = df.index.date
    # Tomar la última instancia de cada fecha (el precio de cierre del día)
    df_daily = df.groupby('date').last()
    # Escribir el archivo modificado en la zona Trusted
    df_daily.to_parquet(destination)


for path_tuple in get_s3_paths(stocks_list):
    minutes_to_days(path_tuple)

