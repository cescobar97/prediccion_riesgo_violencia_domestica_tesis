# Predicción de Riesgo Alto en Denuncias de Violencia Doméstica

Este proyecto es una aplicación web de **Streamlit** que predice la probabilidad de riesgo alto en denuncias de violencia doméstica. Utiliza un modelo entrenado previamente para generar predicciones basadas en los datos de entrada proporcionados por el usuario.

## Contenido del Repositorio

Este repositorio contiene los archivos necesarios para ejecutar la aplicación y generar las predicciones:

- **`Análisis Exploratorio de Datos y Desarrollo de Modelo`**: Dentro de esta carpeta se encuentra:
  - **`Desarrollo.ipynb`**: Script que contiene el entrenamiento de distintos modelos, sus respectivas métricas de performance y la elección del mejor en base a estas últimas.
  - **`EDA.ipynb`**: Análisis Exploratorio de Datos de las Denuncias de Violencia Doméstica de la OVD. También se analizan las denuncias de la linea 144.
  - **`requirements_desarrollo_eda.txt`**: Contiene las librerías necesarias para las notebooks de desarrollo y EDA. 
    
  Además en esta carpeta se necesitan:
  - **`denuncias_2017.csv`**: Archivo csv con datos sobre denuncias de violencia doméstica para el año 2017 de la OVD. Este archivo se utiliza para el entrenamiento del modelo.
  - **`denuncias_2018.csv`**: Archivo csv con datos sobre denuncias de violencia doméstica para el año 2018 de la OVD. Este archivo se utiliza para el entrenamiento del modelo.
  - **`denuncias_2019.csv`**: Archivo csv con datos sobre denuncias de violencia doméstica para el año 2019 de la OVD. Este archivo se utiliza para el entrenamiento del modelo.
  - **`denuncias_2020.csv`**: Archivo csv con datos sobre denuncias de violencia doméstica para el año 2020 de la OVD. Este archivo se utiliza para el entrenamiento del modelo.
  - **`denuncias_2021.csv`**: Archivo csv con datos sobre denuncias de violencia doméstica para el año 2021 de la OVD. Este archivo se utiliza para el entrenamiento del modelo.
  - **`denuncias_2022.csv`**: Archivo csv con datos sobre denuncias de violencia doméstica para el año 2022 de la OVD. Este archivo se utiliza para el entrenamiento del modelo.
  - **`denuncias_2023.csv`**: Archivo csv con datos sobre denuncias de violencia doméstica para el año 2023 de la OVD. Este archivo se utiliza para el entrenamiento del modelo.
  - **`personas_2022.csv`**: Archivo csv con datos sobre denuncias de violencia doméstica para el año 2022 de la OVD. Contiene también información del legajo de la denuncia. Este archivo se utiliza para el entrenamiento del modelo.
  - **`personas_2023.csv`**: Archivo csv con datos sobre denuncias de violencia doméstica para el año 2023 de la OVD. Contiene también información del legajo de la denuncia. Este archivo se utiliza para el entrenamiento del modelo.
    
  Estos archivos se pueden descargar desde [aquí](https://datos.csjn.gov.ar/group/violencia-domestica)
  
  - **`linea144-enero-junio-2023.csv`**: Archivo csv con datos sobre denuncias de violencia doméstica de enero a junio 2023 de la linea 144. 

  Este archivo se puede descargar desde [aquí](https://www.argentina.gob.ar/generos/linea-144/datos-publicos-de-la-linea-144-enero-junio-2023)
   
- **`src`**: Dentro de esta carpeta se encuentra:
  - **`modelo_entrenamiento.py`**: Script que contiene el entrenamiento del modelo. Desde este archivo se generan los archivos `model.pkl`, `normalizer.pkl`, `limites_deciles.pkl`, `col_nums.pkl`, `columns.pkl` que se depositan en la carpeta 'model'. Estos archivos ya se encuentran en el repositorio, por lo tanto no es necesario correrlo si no se desea.
    Además en esta carpeta también se necesitan las denuncias en archivo csv.
  - **`requirements_entrenamiento.txt`**: Contiene las librerías necesarias para el script de entrenamiento.

- **`model`**: 
  - Dentro de esta carpeta se encuentra los archivos generados por el script `modelo_produccion`: `model.pkl`, `normalizer.pkl`, `limites_deciles.pkl`, `col_nums.pkl`, `columns.pkl` 
  - **`config.yaml`**: Archivo que contiene los valores de los hiperparámetros necesarios para el entrenamiento del modelo.
    
- **`requirements.txt`**: Archivo que contiene las librerías necesarias para instalar y ejecutar el proyecto.
- **`predict.py`**: Archivo Python que genera las variables necesarias para el modelo, carga el modelos entrenado y genera las predicciones basadas en los datos de entrada.
- **`app.py`**: Archivo que contiene la aplicación de Streamlit que interactúa con el modelo y permite al usuario hacer predicciones.
- **`Dockerfile`**: Contiene los pasos necesarios para crear una imagen Docker para ejecutar la aplicación de Streamlit en un contenedor.


## Requisitos

Para ejecutar este proyecto, **Docker** es la forma más fácil. Se puede descargar en [descargarlo aquí](https://www.docker.com/get-started). No es necesario instalar Python ni las librerías manualmente si se utiliza Docker.

## Instrucciones de uso

### Opción 1

La forma más sencilla de ejecutar esta aplicación es utilizando una imagen ya construida disponible en Docker Hub. No es necesario clonar el repositorio ni instalar dependencias.

1. **Ejecutar los siguientes comandos:**

    ```bash
    docker pull cescobar97/prediccion_riesgo_violencia_domestica:latest
    docker run -p 8501:8501 cescobar97/prediccion_riesgo_violencia_domestica:latest
    ```

2. El paso anterior abrirá la aplicación en el navegador en el puerto 8501, donde se puede interactuar con la aplicación de Streamlit y generar predicciones.



---

### Opción 2

Si fuera necesario construir la imagen de Docker, se deben seguir los siguientes pasos:

1. **Clonar este repositorio:**

    ```bash
    git clone https://github.com/cescobar97/prediccion_riesgo_violencia_domestica_tesis.git
    cd prediccion_riesgo_violencia_domestica_tesis
    ```

2. **Construir la imagen Docker:**

    Asegurarse de estar en la carpeta raíz del repositorio y ejecutar el siguiente comando para construir la imagen Docker:

    ```bash
    docker build -t prediccion_riesgo_violencia_domestica .
    ```

    Este comando utilizará el `Dockerfile` para construir una imagen con todos los componentes necesarios para ejecutar la aplicación.

3. **Ejecutar la aplicación en un contenedor Docker:**

    Una vez que la imagen esté construida, se puede ejecutar el contenedor con el siguiente comando:

    ```bash
    docker run -p 8501:8501 prediccion_riesgo_violencia_domestica
    ```

    Esto abrirá la aplicación en el navegador en el puerto `8501`, donde se puede interactuar con la aplicación de Streamlit y generar predicciones.
