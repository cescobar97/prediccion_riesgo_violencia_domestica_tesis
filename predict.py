#Se importan librerias
import numpy as np
import pandas as pd
import os
import pickle
from catboost import CatBoostClassifier


# Se define directorio de carpeta model
FOLDER = os.path.join(os.getcwd(), 'model') 

# Obtener modelo entrenado y artefactos
def cargar_pickle(folder: str, file_name: str):
    print(f'Cargando {file_name} de manera local')  
    with open(os.path.join(folder, file_name), 'rb') as f:
        fetched_object = pickle.load(f)

    return fetched_object    


def modelo_artefactos():
    print('Cargando archivos pkl')
    normalizer = cargar_pickle(FOLDER, 'normalizer.pkl')
    num_cols = cargar_pickle(FOLDER, 'num_cols.pkl')
    limites_deciles = cargar_pickle(FOLDER, 'limites_deciles.pkl')
    columns = cargar_pickle(FOLDER, 'columns.pkl')
    model = cargar_pickle(FOLDER, 'model.pkl')
    

    return normalizer, num_cols, model, limites_deciles ,columns

# Se generan variables a partir de funciones. Reciben los valores que luego defenirá el usuario
def convertir_a_numero(value):
    """
    Convierte una respuesta tipo 'SI'/'NO' en valor binario (1 o 0).

    Parámetros:
    - value (str): Cadena con el valor 'SI' o cualquier otro valor.

    Retorna:
    - int: 1 si value es 'SI', 0 en caso contrario.
    """
    return 1 if value == 'SI' else 0


def asignar_rangos_edad(edad, denunciada_edad):
    """
    Clasifica la edad de la persona afectada y denunciada en grupos etarios predefinidos.
    Calcula la diferencia entre los grupos etarios asignados.

    Parámetros:
    - edad (int): Edad de la persona afectada.
    - denunciada_edad (str): Grupo etario de la persona denunciada.

    Retorna: grupo_edad, edad_inicio_rango, denunciada_edad_inicio_rango, diferencia_edad.
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
    
    def obtener_grupo_edad(edad):

        if 0 <= edad <= 5:
            return 'NIÑAS/OS ( 0-5 AÑOS)'
        elif 6 <= edad <= 10:
            return 'NIÑAS/OS ( 6-10 AÑOS)'
        elif 11 <= edad <= 14:
            return 'NIÑAS/OS (11-14 AÑOS)'
        elif 15 <= edad <= 17:
            return 'NIÑAS/OS (15-17 AÑOS)'
        elif 18 <= edad <= 21:
            return 'JOVENES (18-21 AÑOS)'
        elif 22 <= edad <= 29:
            return 'ADULTAS (22-29 AÑOS)'
        elif 30 <= edad <= 39:
            return 'ADULTAS (30-39 AÑOS)'
        elif 40 <= edad <= 49:
            return 'ADULTAS (40-49 AÑOS)'
        elif 50 <= edad <= 59:
            return 'ADULTAS (50-59 AÑOS)'
        elif 60 <= edad <= 74:
            return 'MAYORES (60-74 AÑOS)'
        else:
            return 'MAYORES MAS DE 74 AÑOS'
    
    grupo_edad = obtener_grupo_edad(edad)
    edad_inicio_rango = grupo_edad_map.get(grupo_edad, 1000)
    denunciada_edad_inicio_rango = grupo_edad_map.get(denunciada_edad, 1000)
    diferencia_edad = denunciada_edad_inicio_rango - edad_inicio_rango
    return grupo_edad,edad_inicio_rango, denunciada_edad_inicio_rango, diferencia_edad

def contar_violencias(*violencias):
    """
    Cuenta cuántos tipos de violencia fueron marcados como 'SI'.

    Parámetros:
    - *violencias (str): Respuestas individuales ('SI'/'NO') de diferentes tipos de violencia.

    Retorna:
    - int: Cantidad de violencias marcadas como 'SI'.
    """
    return sum(convertir_a_numero(v) for v in violencias)

def calcular_frecuencia_codificada(frecuencia_episodios):
    """
    Traduce la categoría de frecuencia textual a un valor numérico representando cantidad de episodios mensuales.

    Parámetros:
    - frecuencia_episodios (str): Categoría (ej. 'DIARIO', 'SEMANAL', etc.).

    Retorna frecuencia estimada como número
    """

    frecuencia_dias_map = {
        'DIARIO': 30,
        'SEMANAL': 4,
        'QUINCENAL': 2,
        'MENSUAL': 1,
        'ESPORADICO': 0.5,
        'PRIMER EPISODIO': 0.1,
        'NO CORRESPONDE': 0
    }
    frecuencia_codificada = frecuencia_dias_map.get(frecuencia_episodios, 0)
    
    return frecuencia_codificada

def calcular_ponderacion_violencias(violencias):
    """
    Calcula una puntuación ponderada en base a los tipos de violencia presentes.

    Parámetros:
    - violencias (dict): Diccionario con claves como tipos de violencia y valores 1/0 indicando su presencia.

    Retorna la suma de pesos asociados a los tipos de violencia presentes.
    """
    ponderadores = {
        'v_ambiental': 3,
        'v_economica': 1,
        'v_fisica': 6,
        'v_psicologica': 4,
        'v_sexual': 7,
        'v_simbolica': 2,
        'v_social': 5
    }
    return sum(ponderadores.get(var, 0) for var, val in violencias.items() if val == 1)

def calcular_nivel_educacion(nivel_instruccion, denunciada_nivel_instru):

    """
    Convierte los niveles educativos a valores ordinales y calcula la diferencia.

    Parámetros:
    - nivel_instruccion (str): Nivel educativo de la persona afectada.
    - denunciada_nivel_instru (str): Nivel educativo de la persona denunciada.

    Retorna el nivel educativo de la victima y el denunciado como valor numérico y la diferencia educativa
    """

    nivel_educacion_ordinal = {
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
    nivel_instr = nivel_educacion_ordinal.get(nivel_instruccion, 0)
    denunciada_instr = nivel_educacion_ordinal.get(denunciada_nivel_instru, 0)
    return nivel_instr, denunciada_instr, denunciada_instr - nivel_instr

def asignar_comuna(barrio):
    """
    Asigna la comuna correspondiente a un barrio de CABA

    Parámetros:
    - barrio (str): Nombre del barrio.

    Retorna comuna correspondiente según barrio
    """
    barrio_a_comuna = {
        'CABALLITO': 'COMUNA 06',
        'BELGRANO': 'COMUNA 13',
        'SAN NICOLAS': 'COMUNA 01',
        'SAN CRISTOBAL': 'COMUNA 03',
        'FLORESTA': 'COMUNA 10',
        'VILLA URQUIZA': 'COMUNA 12',
        'RETIRO': 'COMUNA 01',
        'BALVANERA': 'COMUNA 03',
        'PALERMO': 'COMUNA 14',
        'BOCA': 'COMUNA 04',  
        'FLORES': 'COMUNA 07',
        'PARQUE AVELLANEDA': 'COMUNA 09',
        'NUEVA POMPEYA': 'COMUNA 04',
        'MATADEROS': 'COMUNA 09',
        'PATERNAL': 'COMUNA 15', 
        'VILLA DEL PARQUE': 'COMUNA 11',
        'LUGANO': 'COMUNA 08', 
        'VILLA CRESPO': 'COMUNA 15',
        'BARRACAS': 'COMUNA 04',
        'VILLA PUEYRREDON': 'COMUNA 12',
        'RECOLETA': 'COMUNA 02',
        'ALMAGRO': 'COMUNA 05',
        'CONSTITUCION': 'COMUNA 01',
        'VILLA DEVOTO': 'COMUNA 11',
        'BOEDO': 'COMUNA 05',
        'PARQUE PATRICIOS': 'COMUNA 04',
        'VILLA REAL': 'COMUNA 10',
        'VILLA SOLDATI': 'COMUNA 08',
        'MONTE CASTRO': 'COMUNA 10',
        'SAAVEDRA': 'COMUNA 12',
        'NUÑEZ': 'COMUNA 13',
        'LINIERS': 'COMUNA 09',
        'VILLA GRAL. MITRE': 'COMUNA 11',
        'PARQUE CHACABUCO': 'COMUNA 07',
        'VILLA ORTUZAR': 'COMUNA 15',
        'SAN TELMO': 'COMUNA 01',
        'PIEDRA BUENA': 'COMUNA 08',  
        'VILLA LURO': 'COMUNA 10',
        'VILLA SANTA RITA': 'COMUNA 11',
        'ONCE': 'COMUNA 03',  
        'MONSERRAT': 'COMUNA 01',
        'COGHLAN': 'COMUNA 12',
        'COLEGIALES': 'COMUNA 13',
        'PUERTO MADERO': 'COMUNA 01',
        'PARQUE CHAS': 'COMUNA 15',
        'CHACARITA': 'COMUNA 15',
        'VERSAILLES': 'COMUNA 10',
        'AGRONOMIA': 'COMUNA 15',
        'VILLA RIACHUELO': 'COMUNA 08',
        'VELEZ SARSFIELD': 'COMUNA 10',
        'OTROS': 'OTRO',
        'OTRO': 'OTRO',
        'SITUACION DE CALLE': 'OTRO'
    }
    return barrio_a_comuna.get(barrio.upper(), None)


def generar_resultados(sample):
    """
    Procesa una muestra individual para extraer variables derivadas necesarias para la predicción

    Parámetros:
    - sample (list): Lista ordenada con valores de entrada 

    Retorna Diccionario con variables derivadas procesadas
    """
    (edad, genero, nacionalidad, barrio, domicilio_provincia, condición_actividad, nivel_instruccion, v_ambiental, v_economica, v_fisica, v_psicologica, v_sexual, v_simbolica, v_social, frecuencia_episodios, denunciada_edad, denunciada_sexo, denunciada_cond_actividad, denunciada_nivel_instru, cohabitan,
    relacion_afectada_denunciada, mes, denuncia_tercera, ingreso, cant_personas_por_legajo, flag_menor, denuncia_con_varon_mayor,denuncia_con_mujer_mayor) = sample

    grupo_edad,edad_inicio_rango, denunciada_edad_inicio_rango, diferencia_edad = asignar_rangos_edad(edad, denunciada_edad)
    cantidad_violencias = contar_violencias(v_ambiental, v_economica, v_fisica, v_psicologica, v_sexual, v_simbolica, v_social)
    frecuencia_codificada = calcular_frecuencia_codificada(frecuencia_episodios)
    nivel_instr, denunciada_instr, diferencia_educacion = calcular_nivel_educacion(nivel_instruccion, denunciada_nivel_instru)
    violencia_dict = {
        'v_ambiental': convertir_a_numero(v_ambiental),
        'v_economica': convertir_a_numero(v_economica),
        'v_fisica': convertir_a_numero(v_fisica),
        'v_psicologica': convertir_a_numero(v_psicologica),
        'v_sexual': convertir_a_numero(v_sexual),
        'v_simbolica': convertir_a_numero(v_simbolica),
        'v_social': convertir_a_numero(v_social)
    }
    ponderacion_violencias = calcular_ponderacion_violencias(violencia_dict)
    violencia_ponderada_frecuencia = ponderacion_violencias * frecuencia_codificada
    comuna = asignar_comuna(barrio)
    mes_barrio = mes+ '_' + barrio


    resultados = {
        'mes_barrio': mes_barrio,
        'edad': edad,
        'genero': genero,
        'nacionalidad': nacionalidad,
        'comuna': comuna,
        'denunciada_edad': denunciada_edad,
        'frecuencia_episodios': frecuencia_episodios,
        'v_ambiental': v_ambiental,
        'v_economica': v_economica,
        'v_fisica': v_fisica,
        'v_psicologica': v_psicologica,
        'v_sexual': v_sexual,
        'v_simbolica': v_simbolica,
        'v_social': v_social,
        'nivel_instruccion': nivel_instr,
        'grupo_edad': grupo_edad,
        'flag_menor': flag_menor,
        'denuncia_con_varon_mayor': denuncia_con_varon_mayor,
        'denuncia_con_mujer_mayor': denuncia_con_mujer_mayor,
        'diferencia_edad': diferencia_edad,
        'cantidad_violencias': cantidad_violencias,
        'violencia_ponderada_frecuencia': violencia_ponderada_frecuencia,
        'denunciada_nivel_instru': denunciada_instr,
        'diferencia_educacion': diferencia_educacion,
        'ponderacion_violencias': ponderacion_violencias,
        'violencia_ponderada_frecuencia': violencia_ponderada_frecuencia,
        'denuncias_frecuencia': cant_personas_por_legajo * frecuencia_codificada,
        'violencia_frecuencia': cantidad_violencias * frecuencia_codificada,
        'cant_personas_por_legajo': cant_personas_por_legajo,
        'cohabitan': cohabitan,
        'condición_actividad': condición_actividad,
        'denunciada_cond_act': denunciada_cond_actividad, 
        'domicilio_provincia': domicilio_provincia,
        'denunciada_sexo': denunciada_sexo,
        'denuncia_tercera': denuncia_tercera,
        'relacion_afectada_denunciada': relacion_afectada_denunciada,
        'ingreso': ingreso
        
}

    return resultados



def predict(sample: list) -> dict:
    """
    Genera la predicción del modelo.
    
    Parámetros: 'sample': Lista

    Retorna un diccionario con los valores ingresados, la probabilidad de riesgo y el decil de riesgo
    """
    # Obtener las nuevas variables de la función generar_nuevas_variables
    nuevas_variables = generar_resultados(sample)

    # Convertir las nuevas variables en un DataFrame
    nuevas_variables_df = pd.DataFrame([nuevas_variables])

    nuevas_variables_df.columns = nuevas_variables_df.columns.str.lower()

    # Normalización y carga del modelo y artefactos
    normalizer, num_cols, model, limites_deciles, columns = modelo_artefactos()

    # Reordenar las columnas para que estén en el mismo orden que el modelo espera
    nuevas_variables_df = nuevas_variables_df[columns]

    # Separar las columnas numéricas de las categóricas antes de la normalización
    columnas_numericas = nuevas_variables_df[num_cols]  

    # Aplicar la normalización solo a las columnas numéricas
    nuevas_variables_df[num_cols] = normalizer.transform(columnas_numericas)

    # Probabilidad
    y_proba = model.predict_proba(nuevas_variables_df)
    probabilidad_clase = y_proba[0][1]

    # Mostrar las probabilidades
    print(f'y_probabilities: {y_proba}')

    # Calcular el decil de riesgo según la probabilidad
    predicted_decile = np.digitize(probabilidad_clase, limites_deciles, right=True)
    if predicted_decile > 10:
        predicted_decile = 10
    if predicted_decile < 1:
        predicted_decile = 1

    # Resultados a imprimir
    predicted_class = {
        'Valores ingresados': sample,
        'Probabilidad de Riesgo Alto': round(probabilidad_clase * 100),
        'Decil de Riesgo segun Probabilidad Predicha': predicted_decile
    }

    return predicted_class
