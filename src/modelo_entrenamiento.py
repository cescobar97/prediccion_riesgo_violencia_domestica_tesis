# Se importan Librerias

import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import pickle
import yaml
import warnings
warnings.filterwarnings("ignore")


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier

import os


# Se importan csv con datos de denuncias

denuncias_caract_2023 = pd.read_csv('data/personas_2023.csv',sep=';') 

denuncias_caract_2022 = pd.read_csv('data/personas_2022.csv',sep=';')

denuncias_2023 = pd.read_csv('data/denuncias_2023.csv',sep=';')

denuncias_2022 = pd.read_csv('data/denuncias_2022.csv',sep=';')

denuncias_2021 = pd.read_csv('data/denuncias_2021.csv',sep=',')

denuncias_2020 = pd.read_csv('data/denuncias_2020.csv',sep=';')

denuncias_2019 = pd.read_csv('data/denuncias_2019.csv',sep=';')

denuncias_2018 = pd.read_csv('data/denuncias_2018.csv',sep=';')

denuncias_2017 = pd.read_csv('data/denuncias_2017.csv',sep=';')



# Se definen rutas para guardar los archivos

# Ruta base al proyecto
BASE_DIR = os.getcwd()  

# Construir la ruta a la carpeta 'model'
FOLDER = os.path.join(BASE_DIR, 'model')

#### Procesamiento de los datos

print('Script para correr un modelo Catboost que predice niveles de riesgo en denuncias de violencia doméstica')

print('A lo largo del script se generan y guardan files necesarios para el funcionamiento de la aplicación que se genera posteriormente')

def procesar(denuncias_2017, denuncias_2018, denuncias_2019, denuncias_2020, 
             denuncias_2021, denuncias_2022, denuncias_2023, 
             denuncias_caract_2022, denuncias_caract_2023):
    

    """
    Procesa CSV con las denuncias de cada año.

    Parámetros:
    - denuncias_2017 a denuncias_2023: DataFrames con las denuncias de cada año.
    - denuncias_caract_2022, denuncias_caract_2023: DataFrames con características adicionales de los años 2022 y 2023

    Retorna un DataFrame unificado con las denuncias de cada año y las columnas en común
    """

    # Se crea una lista de dataframes
    dataframes = [denuncias_2017, denuncias_2018, denuncias_2019, denuncias_2020, 
                  denuncias_2021, denuncias_2022, denuncias_2023, 
                  denuncias_caract_2022, denuncias_caract_2023]
    
    for df in dataframes:
        df.columns = [col.lower() for col in df.columns]
        
    # Seleccionar columnas de las bases de características
    denuncias_caract_2023 = denuncias_caract_2023[['id_datos_abiertos_legajo', 'ingreso', 'denuncia_tercera']]
    denuncias_caract_2022 = denuncias_caract_2022[['id_datos_abiertos_legajo', 'ingreso', 'denuncia_tercera']]
    
    # Merge con las tablas de características
    denuncias_2023 = denuncias_2023.merge(denuncias_caract_2023, on='id_datos_abiertos_legajo', how='left')
    denuncias_2022 = denuncias_2022.merge(denuncias_caract_2022, on='id_datos_abiertos_legajo', how='left')

    # Renombrar las columnas para que coincidan entre las bases de datos
    nuevos_nombres = {
        'viol_ambient': 'v_ambiental',
        'viol_fisica': 'v_fisica',
        'viol_econ': 'v_economica',
        'viol_psicol': 'v_psicologica',
        'viol_sexual': 'v_sexual',
        'viol_simb': 'v_simbolica',
        'viol_social': 'v_social',
        'legajo_tercera': 'denuncia_tercera'
    }
    
    nuevos_nombres2 = {
        'v_ambient': 'v_ambiental',
        'v_fisica': 'v_fisica',
        'v_econ': 'v_economica',
        'v_psicol': 'v_psicologica',
        'v_sexual': 'v_sexual',
        'v_simb': 'v_simbolica',
        'v_social': 'v_social',
        'denunciada_nivel instru': 'denunciada_nivel_instru',
        'leg_tercera': 'denuncia_tercera'
    }

    nuevos_nombres3 = {
        'v_amb': 'v_ambiental',
        'v_fisica': 'v_fisica',
        'v_econ': 'v_economica',
        'v_psic': 'v_psicologica',
        'v_sex': 'v_sexual',
        'v_simb': 'v_simbolica',
        'v_soc': 'v_social',
        'evaluacion_riesgo': 'eva_riesgo',
        'condicion_actividad': 'condición_actividad',
        'sexo_genero': 'genero',
        'denunciada_cond_lab': 'denunciada_cond_act',
        'nivel_instru': 'nivel_instruccion',
        'condición_laboral': 'condición_actividad',
        'denunciada_sexo_genero': 'denunciada_sexo',
        'leg_tercera': 'denuncia_tercera'
    }

    nuevos_nombres4 = {
        'v_amb': 'v_ambiental',
        'v_fisica': 'v_fisica',
        'v_econ': 'v_economica',
        'v_psic': 'v_psicologica',
        'v_sex': 'v_sexual',
        'v_simb': 'v_simbolica',
        'v_soc': 'v_social',
        'evaluacion_riesgo': 'eva_riesgo',
        'condicion_actividad': 'condición_actividad',
        'sexo_genero': 'genero',
        'denunciada_cond_lab': 'denunciada_cond_act',
        'nivel_instru': 'nivel_instruccion',
        'condición_laboral': 'condición_actividad',
        'denunciada_sexo_genero': 'denunciada_sexo',
        'sexo': 'genero',
        'leg_tercera': 'denuncia_tercera'
    }

    nuevos_nombres5 = {'leg_tercera': 'denuncia_tercera',
                       'v_económica':'v_economica'}

    # Renombrar las columnas en cada DataFrame
    denuncias_2017 = denuncias_2017.rename(columns=nuevos_nombres3)
    denuncias_2018 = denuncias_2018.rename(columns=nuevos_nombres3)
    denuncias_2019 = denuncias_2019.rename(columns=nuevos_nombres4)
    denuncias_2020 = denuncias_2020.rename(columns=nuevos_nombres3)
    denuncias_2021 = denuncias_2021.rename(columns=nuevos_nombres)
    denuncias_2023 = denuncias_2023.rename(columns=nuevos_nombres2)
    denuncias_2022 = denuncias_2022.rename(columns=nuevos_nombres5)

    # Seleccionar columnas comunes entre los DataFrames
    columnas_comunes = [
        'año', 'barrio', 'categoria_ocupacional', 'categoria_ocupacional_detalle', 'cohabitan', 
        'comuna', 'condición_actividad', 'denunciada_cond_act', 'denunciada_edad', 'denunciada_nivel_instru', 
        'denunciada_sexo', 'domicilio_provincia', 'edad', 'eva_riesgo', 'frecuencia_episodios', 'genero', 
        'grupo_edad', 'id_datos_abiertos_legajo', 'id_datos_abiertos_persona', 'ingreso', 'localidad_otras_provincias', 
        'mes', 'nacionalidad', 'nivel_instruccion', 'relacion_afectada_denunciada', 'v_ambiental', 'v_economica', 
        'v_fisica', 'v_psicologica', 'v_sexual', 'v_simbolica', 'v_social', 'denuncia_tercera'
    ]
    
    # Filtrar las columnas comunes en cada DataFrame
    denuncias_2017 = denuncias_2017[columnas_comunes]
    denuncias_2018 = denuncias_2018[columnas_comunes]
    denuncias_2019 = denuncias_2019[columnas_comunes]
    denuncias_2020 = denuncias_2020[columnas_comunes]
    denuncias_2021 = denuncias_2021[columnas_comunes]
    denuncias_2022 = denuncias_2022[columnas_comunes]
    denuncias_2023 = denuncias_2023[columnas_comunes]

    # Asegurarse de que las variables numéricas no generen errores
    denuncias_2019['edad'] = pd.to_numeric(denuncias_2019['edad'], errors='coerce')
    denuncias_2019['edad'] = denuncias_2019['edad'].fillna(1000).astype(int)
    
    # Definir el tipo de variables para todas base 2019 ya que es distinta al resto
    denuncias_2019 = denuncias_2019.astype({
        'año': 'str', 'barrio': 'str', 'categoria_ocupacional': 'str', 
        'categoria_ocupacional_detalle': 'str', 'cohabitan': 'str', 
        'comuna': 'str', 'condición_actividad': 'str', 'denunciada_cond_act': 'str',
        'denunciada_edad': 'str', 'denunciada_nivel_instru': 'str', 
        'denunciada_sexo': 'str', 'denuncia_tercera': 'str', 
        'domicilio_provincia': 'str', 'edad': 'int64', 'eva_riesgo': 'str', 
        'frecuencia_episodios': 'str', 'genero': 'str', 'grupo_edad': 'str', 
        'id_datos_abiertos_legajo': 'str', 'id_datos_abiertos_persona': 'str', 
        'ingreso': 'str', 'localidad_otras_provincias': 'str', 'mes': 'str', 
        'nacionalidad': 'str', 'nivel_instruccion': 'str', 'relacion_afectada_denunciada': 'str',
        'v_ambiental': 'str', 'v_economica': 'str', 'v_fisica': 'str', 'v_psicologica': 'str', 
        'v_sexual': 'str', 'v_simbolica': 'str', 'v_social': 'str'
    })

    # Concatenar todos los DataFrames
    dataframes = [denuncias_2017, denuncias_2018, denuncias_2019, denuncias_2020, 
                  denuncias_2021, denuncias_2022, denuncias_2023]

    df = pd.concat(dataframes, ignore_index=True)
    
    # Eliminar columnas innecesarias
    df = df.drop(['categoria_ocupacional', 'categoria_ocupacional_detalle', 'localidad_otras_provincias', 'año'], axis=1)

    return df

df=procesar(denuncias_2017, denuncias_2018, denuncias_2019, denuncias_2020, 
                               denuncias_2021, denuncias_2022, denuncias_2023, 
                               denuncias_caract_2022, denuncias_caract_2023 
                              )



### 2. Limpieza de datos y generacion de variables nuevas

def estandarizar_nombres_y_nulos(df):
    """
    Estandariza los nombres de columnas y unifica valores nulos en un DataFrame.

    Parámetros:
        df: DataFrame con los datos originales.

    Retorna:
        DataFrame con los nombres de columnas en minúsculas y valores nulos comunes reemplazados por np.nan,
        excluyendo las columnas identificadoras 'id_datos_abiertos_legajo' y 'id_datos_abiertos_persona'.
    """
    # Convertir nombres de columnas a minúsculas
    df.columns = df.columns.str.lower()
    
    # Columnas identificadoras
    columnas_excluir = ['id_datos_abiertos_legajo', 'id_datos_abiertos_persona']
    
    # Valores considerados como nulos
    valores_nulos = ['#NULL!', 'nan', '.', 'SIN DATOS', 'DESCONOCE','NaN']
    
    # Reemplazar en cada columna (excepto las excluidas)
    for col in df.columns:
        if col not in columnas_excluir:
            df[col] = df[col].replace(valores_nulos, np.nan)

    return df 


df=estandarizar_nombres_y_nulos(df)


def limpiar_y_transformar_edad(df):
    """
    Limpia y transforma la columna 'edad'.

    Parámetros:
        df: DataFrame con una columna 'edad'

    Retorna:
        DataFrame con la columna 'edad' convertida a entero y los valores nulos reemplazados por la media
        de edades válidas.
    """

    # Rellenar nulos con valor fuera de rando para evitar errores en el casteo
    df['edad'] = df['edad'].fillna(1000)

    # Convertir a int 
    df['edad'] = df['edad'].astype(int)

    # Calcular la media excluyendo valores no validos
    edad_media = df.loc[df['edad'] != 1000, 'edad'].mean()

    # Reemplazar valores no validos por la media 
    df.loc[df['edad'] == 1000, 'edad'] = int(round(edad_media))

    return df


df=limpiar_y_transformar_edad(df)


def estandarizar_ingreso(df):
    """
    Estandariza las categorías de la columna 'ingreso'.

    Parámetros:
        df: DataFrame con una columna 'ingreso'.

    Retorna:
        DataFrame con la columna 'ingreso' estandarizada en categorías comunes 
    """
    # Mapea de valores originales a categorías estandarizadas
    mapeo_ingreso = {
        'COMISARÍA DE LA MUJER': 'COMISARIA',
        'POLICIA DE LA CIUDAD': 'FUERZAS DE SEGURIDAD',
        'GENDARMERIA': 'FUERZAS DE SEGURIDAD',
        'PREFECTURA': 'FUERZAS DE SEGURIDAD',
        'POLICIA METROPOLITANA': 'FUERZAS DE SEGURIDAD',
        'PROFESIONAL DE LA ABOGACIA': 'ABOGADA/O',
        'PATROCINIO JURIDICO GRATUITO': 'ABOGADA/O',
        'PROGRAMA VICTIMAS CONTRA LAS VIOLENCIAS ': 'PROGRAMAS',
        'PROGRAMA PROTEGER': 'PROGRAMAS',
        'DENUNCIA ANTERIOR': 'DENUNCIA ANTERIOR',
        'REFERENCIAS DE OTRAS PERSONAS': 'REFERENCIAS DE OTRAS PERSONAS',
        'SISTEMA DE SALUD': 'SISTEMA DE SALUD',
        'SISTEMA EDUCATIVO': 'SISTEMA EDUCATIVO',
        'LINEA 144': 'LINEA 144',
        'JUSTICIA CIVIL': 'JUSTICIA',
        'JUSTICIA PENAL CONTRAVENCIONAL Y DE FALTAS': 'JUSTICIA',
        'JUSTICIA PENAL DE INSTRUCCION': 'JUSTICIA',
        'JUSTICIA PENAL CORRECCIONAL': 'JUSTICIA',
        'JUSTICIA PENAL DE MENORES': 'JUSTICIA',
        'ATENCION ANTERIOR EN OVD': 'OVD',
        'CONSULTA ANTERIOR EN OVD': 'OVD',
        '137 BRIGADA': '137 LINEA',
        ' 137 BRIGADA': '137 LINEA',
        ' 137 LINEA': '137 LINEA',
        '137 BRIGADA EN OVD': '137 LINEA',
        ' 137 BRIGADA EN OVD': '137 LINEA'
    }

    # Pasar todo a mayúsculas
    df['ingreso'] = df['ingreso'].str.upper()

    # Reemplazar según el diccionario
    df['ingreso'] = df['ingreso'].replace(mapeo_ingreso)

    # Categorías esperadas después del mapeo
    categorias_validas = [
        'COMISARIA', 'FUERZAS DE SEGURIDAD', 'ABOGADA/O', 'PROGRAMAS',
        'DENUNCIA ANTERIOR', 'REFERENCIAS DE OTRAS PERSONAS', 'SISTEMA DE SALUD',
        'SISTEMA EDUCATIVO', 'LINEA 144', 'JUSTICIA', 'OVD', '137 LINEA'
    ]

    # Asignar 'OTROS' a lo que no esté en el listado
    df.loc[~df['ingreso'].isin(categorias_validas), 'ingreso'] = 'OTROS'

    return df


df=estandarizar_ingreso(df)


def asignar_grupo_edad(df):
    """
    Asigna un grupo etario basado en la columna 'edad'. Si bien ya existe se asegura que esten bien asignadas.

    Parámetros:
        df: DataFrame que contiene la columna 'edad'.

    Retorna:
        DataFrame con una columna actualizada 'grupo_edad' que clasifica las edades en rangos etarios definidos.

    """
    def clasificar_grupo_edad(edad):
        if pd.isna(edad):
            return None
        if 0 <= edad <= 5:
            return "NIÑAS/OS ( 0-5 AÑOS)"
        elif 6 <= edad <= 10:
            return "NIÑAS/OS ( 6-10 AÑOS)"
        elif 11 <= edad <= 14:
            return "NIÑAS/OS (11-14 AÑOS)"
        elif 15 <= edad <= 17:
            return "NIÑAS/OS (15-17 AÑOS)"
        elif 18 <= edad <= 21:
            return "JOVENES (18-21 AÑOS)"
        elif 22 <= edad <= 29:
            return "ADULTAS (22-29 AÑOS)"
        elif 30 <= edad <= 39:
            return "ADULTAS (30-39 AÑOS)"
        elif 40 <= edad <= 49:
            return "ADULTAS (40-49 AÑOS)"
        elif 50 <= edad <= 59:
            return "ADULTAS (50-59 AÑOS)"
        elif 60 <= edad <= 74:
            return "MAYORES (60-74 AÑOS)"
        elif edad > 74:
            return "MAYORES MAS DE 74 AÑOS"
        else:
            return "Edad fuera de rango"
    
    df["grupo_edad"] = df["edad"].apply(clasificar_grupo_edad)

    return df


df=asignar_grupo_edad(df)


def estandarizar_categorias(df):
    """
    Corrige y estandariza valores en columnas categóricas según reglas predefinidas.

    Parámetros:
        df: DataFrame con columnas categóricas que necesitan limpieza y estandarización.

    Retorna:
        DataFrame con valores corregidos y estandarizados en las columnas categóricas.
    """

    # denunciada_cond_act
    df['denunciada_cond_act'] = df['denunciada_cond_act'].replace({
        'TRABAJA SIN REMUNERACIÓN-AMA DE CASA': 'TRABAJA SIN REMUNERACION / AMA DE CASA',
        'TRABAJA SIN REMUNERACI�N-AMA DE CASA': 'TRABAJA SIN REMUNERACION / AMA DE CASA',
        'TRABAJA SIN REMUNERACION - AMA DE CASA-': 'TRABAJA SIN REMUNERACION / AMA DE CASA',
        'JUBILADA / PENSIONADA ': 'JUBILADA/PENSIONADA',
        'JUBILADA / PENSIONADA': 'JUBILADA/PENSIONADA'
    })
    # condición_actividad
    df['condición_actividad'] = df['condición_actividad'].replace({
     'TRABAJA SIN REMUNERACION - AMA DE CASA-': 'TRABAJA SIN REMUNERACION / AMA DE CASA',
     'JUBILADA / PENSIONADA': 'JUBILADA/PENSIONADA'
    })

    # denunciada_edad
    df['denunciada_edad'] = df['denunciada_edad'].replace({
        'MAYORES MAS DE 74 AÑO': 'MAYORES MAS DE 74 AÑOS',
        'NIÑAS/OS (15-17 AÑOS': 'NIÑAS/OS (15-17 AÑOS)',
        'NIÑAS/OS (11-14 AÑOS': 'NIÑAS/OS (11-14 AÑOS)'
    })

    # denunciada_sexo
    df['denunciada_sexo'] = df['denunciada_sexo'].replace({
        'VARON': 'MASCULINO',
        'MUJER': 'FEMENINO'
    })

    # genero
    df['genero'] = df['genero'].replace({
        'VARON': 'MASCULINO',
        'MUJER': 'FEMENINO',
        'MUJER CIS': 'FEMENINO',
        'VARON CIS': 'MASCULINO'
    })

    # domicilio_provincia
    df.loc[~df['domicilio_provincia'].isin(['CIUDAD DE BS. AS.', 'BUENOS AIRES']), 'domicilio_provincia'] = 'OTROS'

    # nacionalidad
    df.loc[df['nacionalidad'].isin(['PARAGUAY', 'URUGUAY', 'CHILE', 'BOLIVIA', 'BRASIL']), 'nacionalidad'] = 'PAIS LIMITROFE'
    df.loc[(df['nacionalidad'] != 'ARGENTINA') & (df['nacionalidad'] != 'PAIS LIMITROFE'), 'nacionalidad'] = 'OTROS'

    # nivel_instruccion
    df['nivel_instruccion'] = df['nivel_instruccion'].replace({
        'SIN INSTRUCCIÓN': 'SIN INSTRUCCION',
        'SIN ESCOLARIZAR': 'SIN INSTRUCCION'
    })

    # denunciada_nivel_instru
    df['denunciada_nivel_instru'] = df['denunciada_nivel_instru'].replace({
        'SECUNADRIO INCOMPLETO': 'SECUNDARIO INCOMPLETO',
        'SIN ESCOLARIZAR': 'SIN INSTRUCCION',
        'SIN INSTRUCCIÓN': 'SIN INSTRUCCION',
        'NIVEL INICIAL -JARDIN-': np.nan
    })

    # relacion_afectada_denunciada
    df['relacion_afectada_denunciada'] = df['relacion_afectada_denunciada'].replace({
        'OTROS ': 'OTROS',
        'OTRO': 'OTROS',
        'PAREJA CONVIVIENTE' : 'CONVIVIENTES'
    })

    # barrio y comuna
    df.loc[df['barrio'].isnull() & (df['domicilio_provincia'] != 'CIUDAD DE BS. AS.'), 'barrio'] = 'OTRO'
    df.loc[df['comuna'].isnull() & (df['domicilio_provincia'] != 'CIUDAD DE BS. AS.'), 'comuna'] = 'OTRO'

    return df


df= estandarizar_categorias(df)    

#
def rellenar_nulos_por_moda(df, columna_objetivo, columna_grupo):
    """
   Rellena valores nulos en una columna usando la moda del grupo correspondiente.

    Parámetros:
        df: DataFrame con los datos.
        columna_objetivo: nombre de la columna donde se rellenarán los nulos.
        columna_grupo: nombre de la columna que define los grupos para calcular la moda.
       

    Retorna:
        DataFrame con los valores nulos en `columna_objetivo` reemplazados por la moda de su grupo.
    """

    # Calculamos la moda por grupo
    moda_por_grupo = (
        df.groupby(columna_grupo)[columna_objetivo]
          .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
    )

    def rellenar_ignorando_nan(row):
        if pd.isna(row[columna_objetivo]):
            grupo = row[columna_grupo]
            # Si el grupo es NaN, no rellenar, devolver el valor original (NaN)
            if pd.isna(grupo):
                return row[columna_objetivo]
            # Si grupo válido, devolver la moda para ese grupo o el valor original si no hay moda
            return moda_por_grupo.get(grupo, row[columna_objetivo])
        else:
            return row[columna_objetivo]

    
    df[columna_objetivo] = df.apply(rellenar_ignorando_nan, axis=1)
    
    return df


# Rellenar nivel_instruccion
df = rellenar_nulos_por_moda(df, 'nivel_instruccion','grupo_edad')

# Rellenar denunciada_nivel_instru 
df = rellenar_nulos_por_moda(df, 'denunciada_nivel_instru','denunciada_edad')

# Rellenar condición_actividad
df = rellenar_nulos_por_moda(df, 'condición_actividad','grupo_edad')

# Rellenar denunciada_cond_act
df = rellenar_nulos_por_moda(df, 'denunciada_cond_act','denunciada_edad')

#
# Se mantienen los casos en los que hay datos sobre la convivencia, dado que el porcentaje de casos que no hay datos es bajo
df = df[df['cohabitan'] != 'S/D']

# Se mantienen los casos en los que hay datos sobre la variable objetivo
df = df[df['eva_riesgo'].notnull()]

# Para las columnas con una pequeño porcentaje de valores nulos, se rellena con la moda
for col in df.columns:
    if df[col].isnull().any():
        moda = df[col].mode()[0]
        df[col] = df[col].fillna(moda)

# Se generan nuevas variables
#### Se generan nuevas variables

# Las variables que se generan a continuación, se idearon a partir de la lectura de bibliografía al respecto y las variables disponibles en el dataset

def flag_menores(df):
    """
    Crea una columna 'flag_menor' que indica si un legajo tiene al menos una persona menor de 18 años.

    Parámetros:
        df: DataFrame con las columnas 'id_datos_abiertos_legajo' y 'edad'.

    Retorna:
        DataFrame con una nueva columna 'flag_menor' con valores 1 si existe alguna persona menor de 18 
        en ese legajo, o 0 en caso contrario.
    """
    df['flag_menor'] = df.groupby('id_datos_abiertos_legajo')['edad'] \
                          .transform(lambda x: (x < 18).any()).astype(int)
    return df


def flag_mayores_por_genero(df, genero: str, flag_name: str):
    """
    Crea un flag para legajos que tengan al menos una persona del género especificado con edad entre 18 y 60 años.

    Parámetros:
        df: DataFrame con las columnas 'id_datos_abiertos_legajo', 'edad' y 'genero'.
        genero: string indicando el género a filtrar (ej. 'MASCULINO', 'FEMENINO').
        flag_name: nombre de la columna flag que se creará.

    Retorna:
        DataFrame con una nueva columna flag que indica con 1 si el legajo tiene personas del género y rango de edad indicados, 0 caso contrario.
    """
    mask = (df['edad'] >= 18) & (df['edad'] <= 60) & (df['genero'] == genero)
    ids = df.loc[mask, 'id_datos_abiertos_legajo'].unique()
    df[flag_name] = df['id_datos_abiertos_legajo'].isin(ids).astype(int)
    return df


def convertir_flag_a_si_no(df, columnas):
    """
    Convierte columnas con valores 0/1 a valores 'SI'/'NO'.

    Parámetros:
        df: DataFrame que contiene las columnas a convertir.
        columnas: lista de nombres de columnas que tienen valores 0/1.

    Retorna:
        DataFrame con las columnas indicadas convertidas a valores 'SI' para 1 y 'NO' para 0.
    """
    for col in columnas:
        df[col] = df[col].map({1: 'SI', 0: 'NO'})
    return df

df = flag_menores(df)
df = flag_mayores_por_genero(df, 'MASCULINO', 'denuncia_con_varon_mayor')
df = flag_mayores_por_genero(df, 'FEMENINO', 'denuncia_con_mujer_mayor')
df = convertir_flag_a_si_no(df, ['flag_menor','denuncia_con_varon_mayor', 'denuncia_con_mujer_mayor'])


# Crear la columna cant_personas_por_legajo
df['cant_personas_por_legajo'] = df.groupby('id_datos_abiertos_legajo')['id_datos_abiertos_legajo'].transform('count')


def asignar_rangos_edad(df):
    """
    Asigna valores numéricos a partir del grupo de edad para facilitar cálculos.

    Parámetros:
        df (DataFrame): debe contener las columnas 'grupo_edad' y 'denunciada_edad'

    Retorna:
        DataFrame con nuevas columnas 'edad_inicio_rango', 'denunciada_edad_inicio_rango' y 'diferencia_edad'
    """
    grupo_edad_map = {
        'ADULTAS (30-39 AÑOS)': 30,
        'ADULTAS (22-29 AÑOS)': 22,
        'ADULTAS (40-49 AÑOS)': 40,
        'NIÑAS/OS ( 0-5 AÑOS)': 0,
        'NIÑAS/OS ( 6-10 AÑOS)': 6,
        'ADULTAS (50-59 AÑOS)': 50,
        'NIÑAS/OS (11-14 AÑOS)': 11,
        'JOVENES (18-21 AÑOS)': 18,
        'MAYORES (60-74 AÑOS)': 60,
        'NIÑAS/OS (15-17 AÑOS)': 15,
        'MAYORES MAS DE 74 AÑOS': 75
    }

    df['edad_inicio_rango'] = df['grupo_edad'].map(grupo_edad_map).fillna(1000).astype(int)
    df['denunciada_edad_inicio_rango'] = df['denunciada_edad'].map(grupo_edad_map).fillna(1000).astype(int)
    df['diferencia_edad'] = df['denunciada_edad_inicio_rango'] - df['edad_inicio_rango']

    return df

df = asignar_rangos_edad(df)


def codificar_frecuencia_episodios(df):
    """
    Asigna valores numéricos a categorías de frecuencia de episodios.

    Parámetros:
        df (DataFrame): debe tener la columna 'frecuencia_episodios'

    Retorna:
        DataFrame con nueva columna 'frecuencia_codificada'
    """
    frecuencia_dias = {
        'DIARIO': 30,
        'SEMANAL': 4,
        'QUINCENAL': 2,
        'MENSUAL': 1,
        'ESPORADICO': 0.5,
        'PRIMER EPISODIO': 0.1,
        'NO CORRESPONDE': 0
    }

    df['frecuencia_codificada'] = df['frecuencia_episodios'].map(frecuencia_dias)

    return df

df = codificar_frecuencia_episodios(df)

#
# Generar la columna variable objetivo binaria
df['eva_riesgo_2'] = df['eva_riesgo'].isin(['ALTISIMO', 'ALTO']).astype(int)


def calcular_ponderacion_violencias(df):
    """
    Entrena un modelo de regresión logística para calcular coeficientes de variables de violencia,
    luego asigna un ponderador basado en la importancia de cada tipo y calcula la ponderación total
    por fila.

    Parámetros:
        df (DataFrame): debe contener columnas de variables de violencia (con valores 'SI'/'NO') y 'eva_riesgo_2'

    Retorna:
        df con nueva columna 'ponderacion_violencias'
    """
    violencia_vars = ['v_ambiental', 'v_economica', 'v_fisica', 'v_psicologica', 'v_sexual', 'v_simbolica', 'v_social']

    # Preparar datos
    X = df[violencia_vars].replace({'SI': 1, 'NO': 0})
    y = df['eva_riesgo_2']

    # División de datos para entrenamiento
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=14, test_size=0.3)
 
    # Entrenar modelo
    model = LogisticRegression(solver='lbfgs', max_iter=1000)
    model.fit(X_train, y_train)

    # Coeficientes e importancia
    features = X_train.columns
    coeficientes_crudos = model.coef_.flatten()
    importancia_abs = np.abs(coeficientes_crudos)

    # Crear tabla de importancia y asignar ponderador
    df_importancia = pd.DataFrame({
        'Tipo de Violencia': features,
        'Coeficiente': coeficientes_crudos,
        'Importancia Absoluta': importancia_abs
    }).sort_values(by='Importancia Absoluta', ascending=False).reset_index(drop=True)

    df_importancia['Ponderador'] = len(df_importancia) - df_importancia.index.to_numpy()

    print("\nTabla de ponderadores por tipo de violencia:")
    print(df_importancia[['Tipo de Violencia', 'Coeficiente', 'Ponderador']])

    ponderadores = dict(zip(df_importancia['Tipo de Violencia'], df_importancia['Ponderador']))

    # Calcular ponderación total
    df_violencias_numerico = df[violencia_vars].replace({'SI': 1, 'NO': 0})
    df['ponderacion_violencias'] = df_violencias_numerico.apply(
        lambda row: sum(row[var] * ponderadores.get(var, 1) for var in violencia_vars),
        axis=1
    )

    return df

df=calcular_ponderacion_violencias(df)

# Calcular violencia ponderada por frecuencia
df['violencia_ponderada_frecuencia'] = df['ponderacion_violencias'] * df['frecuencia_codificada']

 ## Calcular frecuencia por cantidad de persona por legajo
df['denuncias_frecuencia'] = df['frecuencia_codificada'] * df['cant_personas_por_legajo']

# Calcular la cantidad de tipo violencias sufridas
    # Contar el número de violencias
violencia_vars = ['v_ambiental', 'v_economica', 'v_fisica', 'v_psicologica', 'v_sexual', 'v_simbolica', 'v_social']
def contar_violencias(row):
    return sum(row[var] == 'SI' for var in violencia_vars)
    
df['cantidad_violencias'] = df[violencia_vars].apply(contar_violencias, axis=1)

# Multiplicar la cantidad de violencias sufridas por la frecuencia codificada
df['violencia_frecuencia'] = df['cantidad_violencias'] * df['frecuencia_codificada']


def crear_variables_educacion_ordinal(df):
    """
    Convierte los niveles de instrucción de denunciada y víctima a variables ordinales,
    y genera una variable que representa la diferencia educativa entre ellas.

    Parámetros:
        df (DataFrame)

    Retorna:
        df con las columnas 'nivel_instruccion', 'denunciada_nivel_instru' convertidas a ordinal,
        y nueva columna 'diferencia_educacion'
    """
    # Mapas para convertir niveles a valores ordinales
    map_denunciada = {
        'SIN INSTRUCCION': 0,
        'PRIMARIO INCOMPLETO': 2,
        'PRIMARIO COMPLETO': 3,
        'SECUNDARIO INCOMPLETO': 4,
        'SECUNDARIO COMPLETO': 5,
        'TERCIARIO INCOMPLETO': 6,
        'TERCIARIO COMPLETO': 7,
        'UNIVERSITARIO INCOMPLETO': 8,
        'UNIVERSITARIO COMPLETO': 9
    }
    map_victima = {
        'SIN INSTRUCCION': 0,
        'NIVEL INICIAL -JARDIN-': 1,
        'PRIMARIO INCOMPLETO': 2,
        'PRIMARIO COMPLETO': 3,
        'SECUNDARIO INCOMPLETO': 4,
        'SECUNDARIO COMPLETO': 5,
        'TERCIARIO INCOMPLETO': 6,
        'TERCIARIO COMPLETO': 7,
        'UNIVERSITARIO INCOMPLETO': 8,
        'UNIVERSITARIO COMPLETO': 9
    }

    # Reemplazos previos para estandarizar nombres
    df['denunciada_nivel_instru'] = df['denunciada_nivel_instru'].replace('SIN ESCOLARIZAR', 'SIN INSTRUCCION')
    df['nivel_instruccion'] = df['nivel_instruccion'].replace('SIN ESCOLARIZAR', 'SIN INSTRUCCION')

    # Mapear valores ordinales
    df['denunciada_nivel_instru'] = df['denunciada_nivel_instru'].map(map_denunciada)
    df['nivel_instruccion'] = df['nivel_instruccion'].map(map_victima)

    # Convertir a entero 
    df['denunciada_nivel_instru'] = df['denunciada_nivel_instru'].astype(int)
    df['nivel_instruccion'] = df['nivel_instruccion'].astype(int)

    # Calcular diferencia educativa
    df['diferencia_educacion'] = df['denunciada_nivel_instru'] - df['nivel_instruccion']

    return df

df = crear_variables_educacion_ordinal(df)

# Se define variable de interaccion entre mes y barrio
df['mes_barrio'] = df['mes'] + '_' + df['barrio']


# Se elimiman variables innecesarias para el entrenamiento del modelo
df = df.drop(columns=['id_datos_abiertos_legajo', 'id_datos_abiertos_persona','edad_inicio_rango','denunciada_edad_inicio_rango','frecuencia_codificada','mes','barrio'])
print(df.columns)

def fit_normalizer(input_data: pd.DataFrame) -> StandardScaler:

    """
    Ajusta un normalizador (StandardScaler) usando las columnas numéricas del DataFrame de entrada. 
    Guarda el objeto del normalizador entrenado y la lista de columnas numéricas utilizadas en archivos binarios (.pkl) 
    para su uso posterior durante la inferencia o nuevas predicciones.

    Parámetro: DataFrame

    Retorna scaler
    """

    # Seleccionar solo columnas numéricas
    num_cols = input_data.select_dtypes(include=['number']).columns  

    # Ajustar el normalizador solo con columnas numéricas
    scaler = StandardScaler()
    print('Se ajusta un Normalizer con el input dado (solo columnas numéricas)')    
    scaler.fit(input_data[num_cols])  

    # Guardar el normalizador y las columnas utilizadas
    file_name = 'normalizer.pkl'
    with open(os.path.join(FOLDER, file_name), 'wb') as f:
        pickle.dump((scaler), f)  

    print(f"Normalizer guardado en: {file_name}")
        
    print('Se guardan nombres de columnas numericas') 
    file_name_cols = 'num_cols.pkl' 
    with open(os.path.join(FOLDER, file_name_cols), 'wb') as f:
        pickle.dump(num_cols, f)
    
    print(f"Nombres de columnas numericas guardado en: {file_name}")
    return scaler

# Se definen variables independientes 
X = df.drop(["eva_riesgo",'eva_riesgo_2'], axis = 1)


# Se define columna target
y=df['eva_riesgo_2']


# Nombres de columnas en minúscula
X.columns = X.columns.str.lower()
# Se definen columnas str
categorical_cols = X.select_dtypes(exclude=['number']).columns.tolist()

# Split entre training y testing 70-30
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=14,test_size=0.3)

# Se definen columnas numéricas
num_cols = X.select_dtypes(include=['number']).columns  

# Ajustar normalizador solo con variables numéricas
normalizer = fit_normalizer(X_train[num_cols])

# Transformar solo las variables numéricas
X_train[num_cols] = normalizer.transform(X_train[num_cols])
X_test[num_cols] = normalizer.transform(X_test[num_cols])


# Se entrena el Modelo

def fit_catboost(X_train, y_train):
    """
    Fit de un Catboost model usando hiperparámetros fijos desde un archivo config.
    Guarda los pesos del modelo en un archivo pickle.

    Parámetros:
    - X_train: DataFrame con las variables independientes del conjunto de entrenamiento (con columnas normalizadas y categóricas sin codificar).
    - y_train: Serie con la variable objetivo.

    Retorna el modelo entrenado
    """

    # Cargar hiperparámetros desde el archivo config.yaml
    file_name = "config.yaml"
    print(f"Se cargan hiperparametros desde: {file_name}")
    with open(os.path.join(FOLDER, file_name), "r") as f:
        config = yaml.safe_load(f)

    params = config["catboost"]  # Obtener los hiperparámetros 

    # Inicializar el modelo con los hiperparámetros cargados
    model = CatBoostClassifier(
        loss_function=params["loss_function"],
        verbose=params["verbose"],
        random_state=params["random_state"],
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        l2_leaf_reg=params["l2_leaf_reg"],
        random_strength=params["random_strength"],
        eval_metric=params["eval_metric"],
        subsample=params["subsample"],
        class_weights=params["class_weights"]
    )

    print('Se entrena un modelo Catboost')  
    # Entrenar el modelo
    model.fit(X_train, y_train, cat_features=categorical_cols)

    file_name = 'columns.pkl'
    with open(os.path.join(FOLDER, file_name), 'wb') as f:
        pickle.dump(X_train.columns.tolist(), f)
    print(f"Nombres de todas las columnas utilizadas en el entrenamiento guardadas en: {file_name}")

    # Guardar el modelo entrenado
    file_name = "model.pkl"
    with open(os.path.join(FOLDER, file_name), "wb") as f:
        pickle.dump(model, f)
    print(f"Se guarda el modelo entrenado en: {file_name}")

    return model
 
model=fit_catboost(X_train, y_train)

# Se evalua el Modelo 
Y_pred_test = model.predict(X_test)
Y_pred_train = model.predict(X_train)
Y_pred_proba = model.predict_proba(X_test)[:, 1]


print("Metricas de performance del modelo en test:")
# Calcular AUC
auc = roc_auc_score(y_test, Y_pred_proba)
print(f'AUC: {auc:.2f}')

# Calcular y mostrar la matriz de confusión
threshold = 0.44
y_pred = np.where(Y_pred_proba >= threshold, 1, 0)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Matriz de Confusión:")
print(conf_matrix)

# Generar Classification Report
report = classification_report(y_test, y_pred)
print("Tabla de clasificación usando un threshold de 0.44:")
print(report)

# Calcular deciles
def calcular_limites_deciles(probabilities: np.array, actual_values: np.array):
    """
    Calcula los límites de los deciles basados en las probabilidades de predicción, 
    y calcula las estadísticas de 'Actual' y 'Lift' para cada decil.
    Al final guarda los decile boundaries en un archivo pickle.
    
    Parámetros:
    'probabilities': Un array de probabilidades de predicción.
    'actual_values': Un array de valores reales o clases.
    'file_name': Ruta del archivo donde se guardarán los límites de los deciles.
    
    Retorna los límites de los deciles.
    """
    # Crear un DataFrame con las probabilidades y valores reales
    results = pd.DataFrame({'Actual': actual_values, 'Probability': probabilities})
    
    # Calcular los límites de los deciles usando np.percentile
    decile_boundaries = np.percentile(probabilities, np.arange(0, 101, 10))  
    
    # Asignar los deciles (grupos por probabilidades)
    results['Decile'] = pd.qcut(results['Probability'], q=10, labels=False)
    
    # Calcular la media de los valores reales por decil
    decile_stats = results.groupby('Decile')['Actual'].mean().reset_index()
    
    # Calcular la tasa de respuesta promedio (para cálculo de Lift)
    average_response_rate = results['Actual'].mean()
    
    # Añadir la columna de Lift
    decile_stats['Lift'] = decile_stats['Actual'] / average_response_rate

    print("Se calculan 10 grupos iguales basados en las probabilidades de prediccion")
    
    # Mostrar las estadísticas por decil 
    print(decile_stats)

    file_name='limites_deciles.pkl'
    # Guardar solo los límites de los deciles en un archivo pickle
    with open(os.path.join(FOLDER, file_name), 'wb') as f:
        pickle.dump(decile_boundaries, f)
    
    print(f"Limites de deciles guardados en: {file_name}")

    return decile_boundaries


deciles=calcular_limites_deciles(Y_pred_proba,y_test)

print("Limites de deciles:", deciles)

print('Fin del script de entrenamiento')