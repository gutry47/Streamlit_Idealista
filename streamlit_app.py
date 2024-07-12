import pandas as pd
import pickle
import pip
import pydeck as pdk
import re
import streamlit as st
import sklearn

# Ruta al archivo CSV
csv_file_path = 'C:/Users/amarz/PycharmProjects/Streamlit_Idealista/data/viviendas_PROD_v2.csv'
# Cargar el archivo CSV en un DataFrame de pandas
anuncios = pd.read_csv(csv_file_path, sep=",")

st.title("TuTecho Search")
st.subheader("Esta herramienta ha sido diseñada como producto personalizado de búsqueda de viviendas de potencial interés de adquisición para TuTecho")

# Crear un panel de filtros en la barra lateral
with st.sidebar:
    st.header("Búsqueda de viviendas")
    st.markdown("### Filtros")

    # Crear un recuadro para los filtros
    with st.form("filter_form"):
        # Filtro por número de habitaciones
        num_habitaciones = st.selectbox(
            "Número de habitaciones", options=[1, 2, 3, 4]
        )

        # Filtro por precio de alquiler
        precio_min, precio_max = st.slider(
            "Precio de alquiler", min_value=0, max_value=3000, value=(0, 3000), step=100
        )

        # Filtro por rentabilidad requerida
        rentabilidad_requerida = st.slider(
            "Rentabilidad requerida (%)", min_value=3, max_value=8, value=3, step=1
        ) / 100

        # Lista de barrios (ordenada alfabéticamente)
        barrios = sorted([
            'Pau de Carabanchel', 'Palacio', 'Malasaña-Universidad', 'Sol', 'Lavapiés-Embajadores', 'Imperial',
            'Huertas-Cortes', 'Acacias', 'Palomeras sureste', 'Portazgo', 'Numancia', 'Ventas', 'Recoletos',
            'Goya', 'Guindalera', 'Prosperidad', 'Palos de Moguer', 'Jerónimos', 'Chueca-Justicia', 'Almagro',
            'Trafalgar', 'Argüelles', 'Arapiles', 'Chopera', 'Delicias', 'Legazpi', 'Pacífico', 'Adelfas',
            'Niño Jesús', 'Pueblo Nuevo', 'Quintana', 'Colina', 'Pinar del Rey', 'Bernabéu-Hispanoamérica',
            'Nueva España', 'Castilla', 'Cuatro Caminos', 'Valdeacederas', 'Berruguete', 'Gaztambide', 'Estrella',
            'Ibiza', 'Castellana', 'Lista', 'Fuente del Berro', 'El Viso', 'Bellas Vistas', 'Sanchinarro',
            'Virgen del Cortijo - Manoteras', 'Butarque', 'Casco Histórico de Vallecas',
            'Ensanche de Vallecas - La Gavia',
            'Nuevos Ministerios-Ríos Rosas', 'Peñagrande', 'Valdezarza', 'La Paz', 'Tres Olivos - Valverde',
            'Mirasierra',
            'Las Tablas', 'Casa de Campo', 'Valdemarín', 'Ciudad Jardín', 'Cuzco-Castillejos', 'Costillares',
            'Ventilla-Almenara', 'Vallehermoso', 'Ciudad Universitaria', 'Campo de las Naciones-Corralejos',
            'Arroyo del Fresno', 'Águilas', 'San Isidro', 'Vista Alegre', 'Puerta Bonita', 'Buena Vista', 'Orcasitas',
            'Entrevías', 'San Diego', 'Marroquina', 'El Pardo', 'El Plantío', 'Fuentelarreina', 'Pilar', 'Montecarmelo',
            'Aravaca', 'Los Cármenes', 'Puerta del Ángel', 'Lucero', 'Aluche', 'Campamento', 'Comillas', 'Moscardó',
            'Opañel', 'Abrantes', 'San Fermín', 'Zofío', 'Fontarrón', 'San Pascual', 'Canillas', 'San Andrés',
            'Los Rosales', 'Cuatro Vientos', '12 de Octubre-Orcasur', 'Pradolongo', 'Almendrales', 'Palomeras Bajas',
            'Pavones', 'Horcajo', 'Vinateros', 'Media Legua', 'Concepción', 'Simancas', 'San Juan Bautista', 'Atalaya',
            'Palomas', 'Conde Orgaz-Piovera', 'Valdebebas - Valdefuentes', 'Apóstol Santiago', 'Timón', 'Arcos',
            'Los Ángeles', 'San Cristóbal', 'Santa Eugenia', 'Casco Histórico de Vicálvaro',
            'Valdebernardo - Valderribas',
            'Ambroz', 'El Cañaveral - Los Berrocales', 'Salvador', 'Hellín', 'Amposta', 'Rosas', 'Canillejas', 'Rejas',
            'Alameda de Osuna', 'Aeropuerto', 'Casco Histórico de Barajas'
        ])
        barrio = st.selectbox("Barrio", barrios)

        # Botón para aplicar filtros
        submit_button = st.form_submit_button(label='Aplicar filtros')

# Solo aplicar los filtros y actualizar el mapa si se ha presionado el botón de aplicar filtros
if submit_button:
    # Filtrar el dataset según los filtros seleccionados
    # Filtrar por número de habitaciones
    anuncios_filtrados = anuncios[anuncios['ROOMNUMBER'] == num_habitaciones]

    # Calcular el rango de precios
    ingresos_anuales_min = precio_min * 12
    ingresos_anuales_max = precio_max * 12
    gastos_anuales_min = ingresos_anuales_min * 0.6
    gastos_anuales_max = ingresos_anuales_max * 0.6
    finalprice_discount_min = (ingresos_anuales_min - gastos_anuales_min) / rentabilidad_requerida
    finalprice_discount_max = (ingresos_anuales_max - gastos_anuales_max) / rentabilidad_requerida

    # Filtrar por rango de precios
    anuncios_filtrados = anuncios_filtrados[
        (anuncios_filtrados['FINALPRICE_DISCOUNT'] >= finalprice_discount_min) &
        (anuncios_filtrados['FINALPRICE_DISCOUNT'] <= finalprice_discount_max)
        ]

    # Filtrar por barrio
    columna_barrio = f"LOCATIONNAME_{barrio.replace(' ', '_')}"
    if columna_barrio in anuncios_filtrados.columns:
        anuncios_filtrados = anuncios_filtrados[anuncios_filtrados[columna_barrio] == 1]

    # Mostrar el mapa con los puntos filtrados
    st.write("Mapa de anuncios filtrados:")
    st.map(anuncios_filtrados[['lat', 'lon']])