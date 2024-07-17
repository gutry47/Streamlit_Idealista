import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import streamlit as st
import folium
from streamlit_folium import folium_static

# Ruta al archivo CSV
csv_file_path = 'C:/Users/amarz/PycharmProjects/Streamlit_Idealista/data/viviendas_PROD_v2.csv'
MOD_file_path = 'C:/Users/amarz/PycharmProjects/Streamlit_Idealista/data/viviendas_MOD_V4.csv'
tutecho_path = 'C:/Users/amarz/PycharmProjects/Streamlit_Idealista/data/tutecho.png'
idealista_path = 'C:/Users/amarz/PycharmProjects/Streamlit_Idealista/data/idealista.png'

# Cargar el archivo CSV en un DataFrame de pandas
anuncios = pd.read_csv(csv_file_path, sep=",")
df = pd.read_csv(MOD_file_path, sep=",")

# Imputar los valores nulos con la mediana de cada columna (excepto las categóricas) - DF
for col in df.select_dtypes(include=[np.number]).columns:
    df[col].fillna(df[col].median(), inplace=True)

for col in anuncios.select_dtypes(include=[np.number]).columns:
    anuncios[col].fillna(anuncios[col].median(), inplace=True)

# Separar la variable objetivo
y = df['FINALPRICE_DISCOUNT']
X = df.drop(columns=['FINALPRICE_DISCOUNT'])

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo XGBoost
xgb_model = XGBRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    colsample_bytree=1.0,
    min_child_weight=1,
    subsample=0.8,
    random_state=42
)

# Entrenar el modelo
xgb_model.fit(X_train, y_train)

# Predecir función para obtener la recomendación de inversión
def obtener_recomendacion_inversion(precio_original, precio_predicho):
    if precio_predicho > precio_original:
        return "SI"  # Inversión recomendable (verde en el mapa)
    else:
        return "NO"  # Mala inversión (rojo en el mapa)

# Título y subtítulo
st.title("TuTecho Search")
st.subheader(
    "Esta herramienta ha sido diseñada como producto personalizado de búsqueda de viviendas de potencial interés de adquisición para TuTecho"
)

# Colocar las imágenes en la barra lateral
st.sidebar.image(tutecho_path, width=60)
st.sidebar.image(idealista_path, width=100)

# Título del panel de filtros
st.sidebar.title("Búsqueda de viviendas")

# Crear un recuadro para los filtros
with st.sidebar.form("filter_form"):
    # Filtro por número de habitaciones
    num_habitaciones = st.selectbox(
        "Número de Habitaciones", options=[1, 2, 3, 4, 5]
    )

    # Filtro por precio de alquiler
    precio_min, precio_max = st.slider(
        "Precio de Alquiler", min_value=0, max_value=4000, value=(0, 4000), step=100
    )

    # Filtro por rentabilidad requerida
    rentabilidad_requerida = st.slider(
        "Rentabilidad Requerida (%)", min_value=3, max_value=8, value=3, step=1
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

# Filtrar el dataset según los filtros seleccionados y almacenarlo en el estado de la sesión
if submit_button:
    if 'ROOMNUMBER' in anuncios.columns:
        anuncios_filtrados = anuncios[anuncios['ROOMNUMBER'] == num_habitaciones]

        ingresos_anuales_min = precio_min * 12
        ingresos_anuales_max = precio_max * 12
        gastos_anuales_min = ingresos_anuales_min * 0.6
        gastos_anuales_max = ingresos_anuales_max * 0.6
        finalprice_discount_min = (ingresos_anuales_min - gastos_anuales_min) / rentabilidad_requerida
        finalprice_discount_max = (ingresos_anuales_max - gastos_anuales_max) / rentabilidad_requerida

        anuncios_filtrados = anuncios_filtrados[
            (anuncios_filtrados['FINALPRICE_DISCOUNT'] >= finalprice_discount_min) &
            (anuncios_filtrados['FINALPRICE_DISCOUNT'] <= finalprice_discount_max)
        ]

        columna_barrio = f"LOCATIONNAME_{barrio.replace(' ', '_')}"
        if columna_barrio in anuncios_filtrados.columns:
            anuncios_filtrados = anuncios_filtrados[anuncios_filtrados[columna_barrio] == 1]

        # Preparar el DataFrame para la predicción
        anuncios_prediccion = anuncios_filtrados.drop(columns=['lon', 'lat', 'ASSETID', 'FINALPRICE_DISCOUNT'])

        # Realizar la predicción con el modelo XGBoost
        predicciones = xgb_model.predict(anuncios_prediccion)

        # Comparar las predicciones con los valores originales de FINALPRICE_DISCOUNT
        anuncios_filtrados['PREDICCION_PRECIO'] = predicciones
        anuncios_filtrados['RECOMENDACION_INVERSION'] = anuncios_filtrados.apply(
            lambda row: obtener_recomendacion_inversion(row['FINALPRICE_DISCOUNT'], row['PREDICCION_PRECIO']),
            axis=1
        )

        # Filtrar las filas según la diferencia porcentual entre precio predicho y precio original
        def diferencia_porcentual(row):
            precio_original = row['FINALPRICE_DISCOUNT']
            precio_predicho = row['PREDICCION_PRECIO']
            return abs(precio_predicho - precio_original) / precio_original <= 0.20

        anuncios_filtrados = anuncios_filtrados[anuncios_filtrados.apply(diferencia_porcentual, axis=1)]

        # Asignar colores según la recomendación de inversión
        anuncios_filtrados['color'] = anuncios_filtrados['RECOMENDACION_INVERSION'].apply(
            lambda x: "#25be4c" if x == "SI" else "#be2528"
        )

        # Calcular la diferencia entre precio actual y precio predicho
        anuncios_filtrados['DIFERENCIA'] = anuncios_filtrados['PREDICCION_PRECIO'] - anuncios_filtrados['FINALPRICE_DISCOUNT']

        # Ordenar el DataFrame
        anuncios_filtrados = anuncios_filtrados.sort_values(
            by=['RECOMENDACION_INVERSION', 'DIFERENCIA'],
            ascending=[False, False]
        )

        # Almacenar los resultados en el estado de la sesión
        st.session_state['anuncios_filtrados_con_prediccion'] = anuncios_filtrados

# Recuperar los datos filtrados del estado de la sesión
anuncios_filtrados_con_prediccion = st.session_state.get('anuncios_filtrados_con_prediccion')

# Crear pestañas para el mapa y la tabla de viviendas filtradas
tabs = ["Mapa de viviendas", "Viviendas filtradas"]
selected_tab = st.radio("Seleccionar vista", tabs)

# Mostrar el mapa de viviendas
if selected_tab == "Mapa de viviendas":
    st.header("Mapa de viviendas filtradas")
    if anuncios_filtrados_con_prediccion is not None and 'lat' in anuncios_filtrados_con_prediccion.columns and 'lon' in anuncios_filtrados_con_prediccion.columns:
        mymap = folium.Map(location=[40.4165, -3.70256], zoom_start=12)
        for index, row in anuncios_filtrados_con_prediccion.iterrows():
            color = row['color']
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=5,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=
                      f"Precio predicción: {row['PREDICCION_PRECIO']:.2f}€<br>"
                      f"Precio actual: {row['FINALPRICE_DISCOUNT']:.2f}€"
            ).add_to(mymap)
        folium_static(mymap)

# Mostrar la tabla de viviendas filtradas
elif selected_tab == "Viviendas filtradas":
    st.header("Viviendas filtradas")
    if anuncios_filtrados_con_prediccion is not None:
        columns_to_display = ['RECOMENDACION_INVERSION', 'FINALPRICE_DISCOUNT', 'PREDICCION_PRECIO'] + [col for col in anuncios_filtrados_con_prediccion.columns if col not in ['RECOMENDACION_INVERSION', 'FINALPRICE_DISCOUNT', 'PREDICCION_PRECIO', 'DIFERENCIA'] and not col.startswith('LOCATIONNAME_')]
        st.dataframe(anuncios_filtrados_con_prediccion[columns_to_display].reset_index(drop=True))
