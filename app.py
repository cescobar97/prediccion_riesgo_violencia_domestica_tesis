#Se importan librerias
import streamlit as st
import predict as pr  
import json
import numpy as np
from matplotlib import colors 
import plotly.graph_objects as go

# Configuración de la aplicación Streamlit
st.title('Predicción de Niveles Altos de Violencia Doméstica')

# Aplicar estilos CSS para cambiar la fuente 
st.markdown("""
    <style>
        /* Cambia la fuente del título principal */
        .stApp h1 {
            font-family: 'Georgia', serif;
            font-weight: bold;
            text-align: center;
        }

        /* Cambia la fuente de los encabezados h3 y h4 */
        .stApp h3, .stApp h4 {
            font-family: 'Georgia', serif;
            font-weight: bold;
            color: #4A4A4A; 
        }

        /* Cambia la fuente de todo el texto en la aplicación */
        html, body, [class*="st-"] {
            font-family: 'Georgia', serif;
        }
    </style>
""", unsafe_allow_html=True)



st.markdown("### ¿De qué se trata esta aplicación?")
st.markdown(
    "Esta aplicación tiene como objetivo **predecir la probabilidad de riesgo** al que está expuesta una persona "
    "tras una denuncia de violencia doméstica, ya sea realizada en primera persona o por un tercero."
)
 

st.markdown("#### ¿Qué se entiende por *riesgo*?")
st.markdown(
    "**Riesgo** se refiere al peligro que afecta la **integridad psicofísica** de una persona como resultado de una situación de violencia."
)

st.markdown("---")  

st.markdown("### ¿Cómo utilizar la aplicación?")
st.markdown(
    "1. Complete **todos los campos** con la información disponible en la denuncia.\n"
    "   - **Datos de la Víctima:** Se recopila información sobre la persona que realizó la denuncia o en nombre de quien se hizo.\n"
    "   - **Tipos y Frecuencia de Violencia Sufrida:** Se identifican las diferentes formas de violencia que ha sufrido la víctima, así como la frecuencia con la que han ocurrido los episodios.\n"
    "   - **Datos del Denunciado:** Se recogen datos sobre la persona acusada, para comprender mejor el contexto de la denuncia.\n"
    "   - **Relación entre la Víctima y el Denunciado:** Se indaga sobre el vínculo entre ambas partes.\n"
    "   - **Datos de la denuncia:** Se registra datos contextuales de la denuncia. Si bien se recopila información a nivel individual, se pueden registrar varias personas bajo el mismo legajo, por ejemplo en el caso de que las víctimas sean un mayor de edad y un menor\n"
    "2. Una vez completado el formulario, presionar el botón 'Predecir' y se calculará la probabilidad y el decil de riesgo. La interpretación de los resultados se presenta junto con los mismos."
)

# El usuario ingresa los valores

st.markdown('### Datos de la Víctima')
edad = st.number_input('Edad de la víctima', step=1)
genero = st.selectbox('Género de la víctima', ['Selecciona una opción','FEMENINO','MASCULINO','MUJER TRANS / TRAVESTI','VARON TRANS','OTROS'])
nacionalidad = st.selectbox('Nacionalidad de la víctima', ['Selecciona una opción', 'ARGENTINA', 'PAIS LIMITROFE','OTROS'])
barrio = st.selectbox('Barrio de domicilio la víctima', ['Selecciona una opción','CABALLITO', 'BELGRANO', 'SAN NICOLAS', 'SAN CRISTOBAL','FLORESTA', 'VILLA URQUIZA', 'RETIRO', 'BALVANERA', 'PALERMO','BOCA', 'FLORES', 'PARQUE AVELLANEDA', 'NUEVA POMPEYA','MATADEROS', 'PATERNAL', 'VILLA DEL PARQUE', 'LUGANO','VILLA CRESPO', 'BARRACAS', 'VILLA PUEYRREDON', 'RECOLETA','ALMAGRO', 'CONSTITUCION', 'VILLA DEVOTO', 'BOEDO','PARQUE PATRICIOS', 'VILLA REAL', 'VILLA SOLDATI', 'MONTE CASTRO','SAAVEDRA', 'NUÑEZ', 'LINIERS', 'VILLA GRAL. MITRE','PARQUE CHACABUCO', 'VILLA ORTUZAR', 'SAN TELMO', 'PIEDRA BUENA','VILLA LURO', 'VILLA SANTA RITA', 'ONCE', 'MONSERRAT', 'COGHLAN','COLEGIALES', 'PUERTO MADERO', 'PARQUE CHAS', 'CHACARITA','VERSAILLES', 'AGRONOMIA', 'VILLA RIACHUELO', 'OTRO','VELEZ SARSFIELD', 'SITUACION DE CALLE'])
with st.expander("Ver detalle"):
    st.markdown("""
**Nota:** En caso de que la persona no resida en la Ciudad Autónoma de Buenos Aires, seleccionar la opción **'OTRO'**.
    """)
domicilio_provincia = st.selectbox('Provincia de domicilio de la víctima', ['Selecciona una opción', 'BUENOS AIRES', 'CIUDAD DE BS. AS.', 'OTROS'])
condición_actividad = st.selectbox('Condición de Actividad de la víctima', ['Selecciona una opción', 'DESOCUPADA', 'INFANCIA', 'JUBILADA/PENSIONADA', 'OCUPADA REMUNERADA', 'OTRA SITUACION', 'TRABAJA SIN REMUNERACION / AMA DE CASA'])
nivel_instruccion = st.selectbox('Máximo nivel de instrucción alcanzado por la víctima', ['Selecciona una opción', 'SIN INSTRUCCION', 'NIVEL INICIAL -JARDIN-', 'PRIMARIO INCOMPLETO', 'PRIMARIO COMPLETO', 'SECUNDARIO INCOMPLETO', 'SECUNDARIO COMPLETO', 'TERCIARIO INCOMPLETO', 'TERCIARIO COMPLETO', 'UNIVERSITARIO INCOMPLETO', 'UNIVERSITARIO COMPLETO'])

st.markdown('#### Tipos y Frecuencia de Violencia Sufrida')
v_ambiental = st.selectbox('¿La víctima sufrió violencia ambiental?', ['Selecciona una opción', 'SI', 'NO'])
v_economica = st.selectbox('¿La víctima sufrió violencia económica?', ['Selecciona una opción', 'SI', 'NO'])
v_fisica = st.selectbox('¿La víctima sufrió violencia física?', ['Selecciona una opción', 'SI', 'NO'])
v_psicologica = st.selectbox('¿La víctima sufrió violencia psicológica?', ['Selecciona una opción', 'SI', 'NO'])
v_sexual = st.selectbox('¿La víctima sufrió violencia sexual?', ['Selecciona una opción', 'SI', 'NO'])
v_simbolica = st.selectbox('¿La víctima sufrió violencia simbólica?', ['Selecciona una opción', 'SI', 'NO'])
v_social = st.selectbox('¿La víctima sufrió violencia social?', ['Selecciona una opción', 'SI', 'NO'])
frecuencia_episodios = st.selectbox('Frecuencia de episodios de violencia', ['Selecciona una opción', 'DIARIO','SEMANAL', 'QUINCENAL', 'MENSUAL', 'ESPORADICO', 'PRIMER EPISODIO', 'NO CORRESPONDE'])

st.markdown('### Datos del Denunciado')
denunciada_edad = st.selectbox('Grupo etario de la persona denunciada', ['Selecciona una opción', 'NIÑAS/OS (11-14 AÑOS)', 'NIÑAS/OS (15-17 AÑOS)', 'JOVENES (18-21 AÑOS)', 'ADULTAS (22-29 AÑOS)', 'ADULTAS (30-39 AÑOS)', 'ADULTAS (40-49 AÑOS)', 'ADULTAS (50-59 AÑOS)', 'MAYORES (60-74 AÑOS)', 'MAYORES MAS DE 74 AÑOS'])
denunciada_sexo = st.selectbox('Sexo del denunciado', ['Selecciona una opción', 'FEMENINO', 'MASCULINO'])
denunciada_cond_actividad = st.selectbox('Condición de actividad de la persona denunciada', ['Selecciona una opción', 'DESOCUPADA', 'INFANCIA', 'JUBILADA/PENSIONADA', 'OCUPADA REMUNERADA', 'OTRA SITUACION', 'TRABAJA SIN REMUNERACION / AMA DE CASA'])
denunciada_nivel_instru = st.selectbox('Máximo nivel de instrucción alcanzado por el denunciado', ['Selecciona una opción', 'SIN INSTRUCCION', 'PRIMARIO INCOMPLETO', 'PRIMARIO COMPLETO', 'SECUNDARIO INCOMPLETO', 'SECUNDARIO COMPLETO', 'TERCIARIO INCOMPLETO', 'TERCIARIO COMPLETO', 'UNIVERSITARIO INCOMPLETO', 'UNIVERSITARIO COMPLETO'])

st.markdown('### Relación entre la Víctima y el Denunciado')
cohabitan = st.selectbox('¿La víctima y el denunciado cohabitan o cohabitaban hasta la semana previa a la denuncia?', ['Selecciona una opción', 'SI', 'NO'])
relacion_afectada_denunciada = st.selectbox('Relación entre la víctima y el denunciado', ['Selecciona una opción', 'AMISTAD', 'CONVIVIENTES', 'CÓNYUGES', 'EX PAREJAS', 'FILIAL', 'FRATERNAL', 'LABORAL', 'NOVIOS', 'OTRO FAMILIAR HASTA 4º GRADO DE PARENTESCO', 'OTROS'])

st.markdown('### Datos de la denuncia')
mes = st.selectbox('Mes realizada la denuncia', ['Selecciona una opción', 'ENERO', 'FEBRERO', 'MARZO', 'ABRIL', 'MAYO', 'JUNIO', 'JULIO', 'AGOSTO', 'SEPTIEMBRE', 'OCTUBRE', 'NOVIEMBRE', 'DICIEMBRE'])
denuncia_tercera = st.selectbox('¿La denuncia fue realizada por una tercera persona?', ['Selecciona una opción', 'SI', 'NO'])
ingreso = st.selectbox('Dependencia, institución o persona que orientó a la persona denunciante a acudir a la OVD', ['Selecciona una opción', '137 LINEA', 'ABOGADA/O', 'COMISARIA', 'DENUNCIA ANTERIOR', 'FUERZAS DE SEGURIDAD', 'JUSTICIA', 'LINEA 144', 'OTROS', 'OVD', 'PROGRAMAS', 'REFERENCIAS DE OTRAS PERSONAS', 'SISTEMA DE SALUD', 'SISTEMA EDUCATIVO'])
with st.expander("Ver detalle"):
    st.markdown("""
- **COMISARIA**: Comisarías de la mujer o de la ciudad.  
- **FUERZAS DE SEGURIDAD**: Policía Metropolitana, Gendarmería, Prefectura, Policía de la Ciudad.  
- **ABOGADA/O**: Profesional de la abogacía o patrocinio jurídico gratuito.  
- **PROGRAMAS**: Programas como "Víctimas contra las violencias" o "Proteger".  
- **JUSTICIA**: Cualquier instancia del Poder Judicial (civil, penal, menores, etc.).  
- **OVD**: Atención o consulta anterior en la Oficina de Violencia Doméstica.  
- **137 LINEA**: Contacto con la Línea 137 o brigadas asociadas.  
- **REFERENCIAS DE OTRAS PERSONAS**: Recomendaciones de terceros no institucionales.  
- **SISTEMA DE SALUD / EDUCATIVO**: Derivación desde hospitales, centros de salud, escuelas, etc.  
- **LINEA 144**: Orientación desde esta línea nacional de atención.  
- **DENUNCIA ANTERIOR**: Existencia de una denuncia previa.  
- **OTROS**: Cualquier otra vía no contemplada arriba.
    """)
cant_personas_por_legajo = st.number_input('Cantidad de personas en el legajo', step=1)
flag_menor = st.selectbox('¿Hay un menor en el legajo? (Menores de 18 años).', ['Selecciona una opción', 'SI', 'NO'])
denuncia_con_varon_mayor = st.selectbox('¿Hay un hombre mayor de edad en el mismo legajo, menor a 61 años?', ['Selecciona una opción', 'SI', 'NO'])
denuncia_con_mujer_mayor = st.selectbox('¿Hay un mujer mayor de edad en el mismo legajo, menor a 61 años?', ['Selecciona una opción', 'SI', 'NO'])

if st.button('Predecir'):
    # Verificar si hay campos vacíos
    campos_vacios = any([
        edad == 0,  
        genero == 'Selecciona una opción',
        nacionalidad == 'Selecciona una opción',
        barrio == 'Selecciona una opción',
        domicilio_provincia == 'Selecciona una opción',
        condición_actividad == 'Selecciona una opción',
        nivel_instruccion == 'Selecciona una opción',
        v_ambiental == 'Selecciona una opción',
        v_economica == 'Selecciona una opción',
        v_fisica == 'Selecciona una opción',
        v_psicologica == 'Selecciona una opción',
        v_sexual == 'Selecciona una opción',
        v_simbolica == 'Selecciona una opción',
        v_social == 'Selecciona una opción',
        frecuencia_episodios == 'Selecciona una opción',
        denunciada_edad == 'Selecciona una opción',
        denunciada_sexo == 'Selecciona una opción',
        denunciada_cond_actividad == 'Selecciona una opción',
        denunciada_nivel_instru == 'Selecciona una opción',
        cohabitan == 'Selecciona una opción',
        relacion_afectada_denunciada == 'Selecciona una opción',
        mes == 'Selecciona una opción',
        denuncia_tercera == 'Selecciona una opción',
        ingreso == 'Selecciona una opción',
        cant_personas_por_legajo == 0,
        flag_menor == 'Selecciona una opción',
        denuncia_con_varon_mayor == 'Selecciona una opción',
        denuncia_con_mujer_mayor == 'Selecciona una opción'
    ])

    if campos_vacios:
        st.warning('⚠️ Por favor, completa todos los campos antes de continuar.')
    else:

        sample = [edad, genero, nacionalidad, barrio, domicilio_provincia, condición_actividad, nivel_instruccion, v_ambiental, v_economica, v_fisica, v_psicologica, v_sexual, v_simbolica, v_social, frecuencia_episodios, denunciada_edad, denunciada_sexo, denunciada_cond_actividad, denunciada_nivel_instru, cohabitan, relacion_afectada_denunciada, mes, denuncia_tercera, ingreso, cant_personas_por_legajo, flag_menor, denuncia_con_varon_mayor, denuncia_con_mujer_mayor
]

        # Se realiza la predicción
        prediction = pr.predict(sample)

        result_json = {"Resultado Obtenido": prediction}

        # Convertir valores numéricos a enteros
        def convertir_int(d):
            for key, value in d.items():
                if isinstance(value, (float, np.float64, int, np.int64)):
                    d[key] = int(value)
                elif isinstance(value, dict):
                    convertir_int(value)
            return d

        result_json = convertir_int(result_json)

        # Extraer valores de predicción
        prediction = result_json['Resultado Obtenido']
        probabilidad = prediction['Probabilidad de Riesgo Alto']
        decil = prediction['Decil de Riesgo segun Probabilidad Predicha']

        # Obtener color en gradiente de rojo a verde
        def gradiente_color(value, min_val=0, max_val=100):
            norm = colors.Normalize(vmin=min_val, vmax=max_val)
            cmap = colors.LinearSegmentedColormap.from_list("my_cmap", ["green", "yellow", "red"])
            color = cmap(norm(value))
            return colors.rgb2hex(color[:3])  

        # Crear plots de termómetro de riesgo
        def crear_termometro(valor, max_val=100, min_val=0):
            color = gradiente_color(valor, min_val, max_val)

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=valor,
                gauge={
                    'axis': {'range': [min_val, max_val]},
                    'bar': {'color': color},  
                    'steps': [
                        {'range': [min_val, valor], 'color': color},
                        {'range': [valor, max_val], 'color': "lightgray"}
                    ]
                }
            ))

            fig.update_layout(
                margin={'t': 50, 'b': 50, 'l': 50, 'r': 50},  
                height=300,  
                showlegend=False
            )

            return fig
        

        def crear_linea_decil(valor, max_val=10, min_val=1):
            fig = go.Figure()

            # Agregar la línea horizontal
            fig.add_trace(go.Scatter(
                x=[min_val, max_val], 
                y=[1, 1],  
                mode="lines",
                line=dict(color="gray", width=5),
                hoverinfo="skip",
                showlegend=False
            ))

            # Agregar el punto que indica el decil
            fig.add_trace(go.Scatter(
                x=[valor], 
                y=[1],  
                mode="markers",
                marker=dict(color="red", size=12),
                textposition="top center",
                hoverinfo="skip",
                showlegend=False
            ))

            fig.update_layout(
                xaxis=dict(title="Decil", range=[min_val, max_val], tickmode="array", tickvals=list(range(min_val, max_val+1)),fixedrange=True),
                yaxis=dict(visible=False, range=[0.5,1.5],fixedrange=True),  
                margin={'t': 50, 'b': 50, 'l': 50, 'r': 50},
                height=150  
            )

            return fig

        # Mostrar gráficos
        st.write("**Gráfico de Probabilidad de Riesgo Alto:**")
        st.write("""
        El siguiente gráfico muestra un termómetro que representa la probabilidad de que la persona esté en una situación de riesgo alto. Este riesgo hace referencia al peligro que podría afectar la **integridad psicofísica** de la persona debido a una situación de violencia. El valor mostrado en el gráfico es un porcentaje, donde los valores más altos indican mayor probabilidad de riesgo.
        """)
        st.plotly_chart(crear_termometro(probabilidad, max_val=100, min_val=0))
        
        st.write("**Gráfico de Decil de Riesgo:**")
        st.write("""
        El siguiente gráfico muestra el **decil de riesgo**, que indica en qué nivel de riesgo se encuentra la persona en comparación con otras situaciones similares. Este cálculo se realiza tomando en cuenta una gran cantidad de denuncias previas, lo que permite poner el caso en contexto y compararlo con otros. El **decil** va del **1 al 10**, donde un número más alto indica un mayor nivel de riesgo. Esta comparación puede ser útil para decidir cómo distribuir los recursos y tomar las mejores decisiones al enfrentar casos similares.
        """)
        st.plotly_chart(crear_linea_decil(decil, max_val=10, min_val=1))
