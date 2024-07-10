import streamlit as st

st.title("TuTecho Search")
st.subheader("Esta herramienta ha sido diseñada como producto personalizado de búsqueda de viviendas de potencial interés de adquisición para TuTecho")
import streamlit as st
import pandas as pd
import pydeck as pdk
import json

# Añadir un título a la barra lateral
st.sidebar.title("Búsqueda de viviendas")

# Crear un contenedor en la barra lateral para agrupar los filtros
with st.sidebar.expander("Filtros de búsqueda", expanded=True):
    # Filtro de arrastrar para indicar el precio de alquiler mensual
    precio_alquiler = st.slider(
        'Precio de alquiler mensual (€)',
        min_value=0,
        max_value=3000,
        value=(500, 1500)
    )

    # Filtro de número de habitaciones con desplegable
    num_habitaciones = st.selectbox(
        'Número de habitaciones',
        [1, 2, 3, 4, 5, 'Más de 5']
    )

    # Filtro de barrio con desplegable
    barrios = ['Centro', 'Norte', 'Sur', 'Este', 'Oeste']
    barrio = st.selectbox(
        'Barrio',
        barrios
    )

# Subir el archivo CSV
#uploaded_file = st.file_uploader("Elige un archivo CSV", type="csv")
uploaded_file = "Data/viviendas_PROD_v1.csv"

if uploaded_file is not None:
    # Leer el archivo CSV subido
    df = pd.read_csv(uploaded_file)

    # Convertir la columna 'geometry' de strings a listas de coordenadas
    try:
        df['geometry'] = df['geometry'].apply(json.loads)
    except json.JSONDecodeError as e:
        st.error(f"Error en la conversión de la columna 'geometry': {e}")

    # Crear el mapa solo si la conversión fue exitosa
    if 'geometry' in df and df['geometry'].apply(lambda x: isinstance(x, list) and len(x) == 2).all():
        # Crear el mapa
        st.title("Mapa de Viviendas en Madrid")
        map_layer = pdk.Layer(
            "ScatterplotLayer",
            data=df,
            get_position="geometry",
            get_radius=100,
            get_fill_color=[255, 0, 0],
            pickable=True
        )

        # Configurar la vista del mapa
        view_state = pdk.ViewState(
            latitude=40.416775,
            longitude=-3.703790,
            zoom=12,
            pitch=50
        )

        # Renderizar el mapa en Streamlit
        st.pydeck_chart(pdk.Deck(
            map_style='mapbox://styles/mapbox/light-v10',
            initial_view_state=view_state,
            layers=[map_layer]
        ))
    else:
        st.error("Los datos de la columna 'geometry' no son válidos.")
