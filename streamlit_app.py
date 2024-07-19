import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import streamlit as st
import folium
from streamlit_folium import folium_static
from sklearn.ensemble import RandomForestRegressor

# Ruta al archivo CSV
csv_file_path = 'C:/Users/amarz/PycharmProjects/Streamlit_Idealista/data/viviendas_PROD_v2.csv'
MOD_file_path = 'C:/Users/amarz/PycharmProjects/Streamlit_Idealista/data/viviendas_MOD_V4.csv'
tutecho = 'C:/Users/amarz/PycharmProjects/Streamlit_Idealista/data/tutecho.png'
idealista = 'C:/Users/amarz/PycharmProjects/Streamlit_Idealista/data/idealista.png'

# Cargar el archivo CSV en un DataFrame de pandas
anuncios = pd.read_csv(csv_file_path, sep=",")
df = pd.read_csv(MOD_file_path, sep=",")

# Imputar los valores nulos con la mediana de cada columna (excepto las categóricas) - DF
for col in df.select_dtypes(include=[np.number]).columns:
    df[col].fillna(df[col].median(), inplace=True)

# Imputar los valores nulos con 0 en lugar de la mediana
for col in anuncios.select_dtypes(include=[np.number]).columns:
    anuncios[col].fillna(0, inplace=True)

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

# Entrenar el modelo XGB
xgb_model.fit(X_train, y_train)

# Crear y entrenar el modelo de Random Forest con los mejores hiperparámetros
best_rf = RandomForestRegressor(
    n_estimators=150,
    max_depth=30,
    max_features=None,
    min_samples_leaf=2,
    min_samples_split=2,
    random_state=42
)

# Entrenar el modelo RF
best_rf.fit(X_train, y_train)

# Función para obtener la recomendación de inversión
def obtener_recomendacion_inversion(precio_original, precio_predicho):
    if precio_predicho > precio_original:
        return "SI"  # Inversión recomendable (verde en el mapa)
    else:
        return "NO"  # Mala inversión (rojo en el mapa)

# Título y subtítulo
st.title("TuTecho Search")
st.markdown(
    '<p style="font-size:16px">Esta aplicación ha sido diseñada como herramienta personalizada de búsqueda de viviendas de potencial interés de adquisición para TuTecho</p>',
    unsafe_allow_html=True
)

# Crear un panel de filtros en la barra lateral
with st.sidebar:
    st.image(tutecho, width=80)
    st.image(idealista, width=80)

    st.header("Búsqueda de viviendas")
    st.markdown("### Filtros")

    # Crear un recuadro para los filtros
    with st.form("filter_form"):
        # Filtro por número de habitaciones
        num_habitaciones = st.selectbox(
            "Número de Habitaciones", options=[1, 2, 3, 4]
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
        submit_button = st.form_submit_button(label='Buscar viviendas')

# Barrios que usan el modelo Random Forest
barrios_rf = ['Opañel', 'Los Ángeles', 'Los Rosales', 'Moscardó', 'Zofío', 'Portazgo']

# Filtrar el dataset según los filtros seleccionados y almacenarlo en el estado de la sesión
if submit_button:
    if 'ROOMNUMBER' in anuncios.columns:
        anuncios_filtrados = anuncios[anuncios['ROOMNUMBER'] == num_habitaciones]

        ingresos_anuales_min = precio_min * 12
        ingresos_anuales_max = precio_max * 12
        gastos_anuales_min = ingresos_anuales_min * 0.15
        gastos_anuales_max = ingresos_anuales_max * 0.15
        finalprice_discount_min = (ingresos_anuales_min - gastos_anuales_min) / rentabilidad_requerida
        finalprice_discount_max = (ingresos_anuales_max - gastos_anuales_max) / rentabilidad_requerida

        anuncios_filtrados = anuncios_filtrados[
            (anuncios_filtrados['FINALPRICE_DISCOUNT'] >= finalprice_discount_min) &
            (anuncios_filtrados['FINALPRICE_DISCOUNT'] <= finalprice_discount_max)
        ]

        # Buscar la columna del barrio seleccionada
        barrio_col = None
        for col in anuncios_filtrados.columns:
            if col.startswith("LOCATIONNAME_") and col.split("LOCATIONNAME_")[1].replace("_", " ") == barrio:
                barrio_col = col
                break

        if barrio_col:
            anuncios_filtrados = anuncios_filtrados[anuncios_filtrados[barrio_col] == 1]

        # Preparar el DataFrame para la predicción
        anuncios_prediccion = anuncios_filtrados.drop(columns=['lon', 'lat', 'ASSETID', 'FINALPRICE_DISCOUNT'])

        # Realizar la predicción con el modelo adecuado
        if barrio in barrios_rf:
            predicciones = best_rf.predict(anuncios_prediccion)
        else:
            predicciones = xgb_model.predict(anuncios_prediccion)

        # Comparar las predicciones con los valores originales de FINALPRICE_DISCOUNT
        anuncios_filtrados['PREDICCION_PRECIO'] = predicciones
        anuncios_filtrados['RECOMENDACION_INVERSION'] = anuncios_filtrados.apply(
            lambda row: obtener_recomendacion_inversion(row['FINALPRICE_DISCOUNT'], row['PREDICCION_PRECIO']),
            axis=1
        )

        # Filtrar según la diferencia porcentual
        anuncios_filtrados = anuncios_filtrados[
            (abs(anuncios_filtrados['PREDICCION_PRECIO'] - anuncios_filtrados['FINALPRICE_DISCOUNT']) /
             anuncios_filtrados['FINALPRICE_DISCOUNT']) <= 0.2
        ]

        # Asignar color según la recomendación de inversión
        anuncios_filtrados['color'] = anuncios_filtrados['RECOMENDACION_INVERSION'].apply(
            lambda x: "#25be4c" if x == "SI" else "#be2528"
        )

        # Guardar los resultados en el estado de la sesión
        st.session_state['anuncios_filtrados_con_prediccion'] = anuncios_filtrados

# Recuperar los datos filtrados del estado de la sesión
anuncios_filtrados_con_prediccion = st.session_state.get('anuncios_filtrados_con_prediccion')

# Diccionario para renombrar las columnas
column_rename_dict = {
    'RECOMENDACION_INVERSION': 'Recomendación de Inversión',
    'FINALPRICE_DISCOUNT': 'Precio Actual',
    'PREDICCION_PRECIO': 'Precio Predicho',
    'ROOMNUMBER': 'Número de Habitaciones',
    'BATHNUMBER': 'Número de Baños',
    'CADCONSTRUCTIONYEAR': 'Año de Construcción',
    'CONSTRUCTEDAREA': 'Metros Cuadrados',
    'DISTANCE_TO_METRO': 'Distancia al Metro',
    'DISTANCE_TO_CASTELLANA': 'Distancia a la Castellana',
    'DISTANCE_TO_CITY_CENTER': 'Distancia al Centro'
}

# Agregar las pestañas para mostrar el mapa y la tabla filtrada
tabs = ["Mapa de viviendas", "Viviendas filtradas"]
selected_tab = st.radio("Seleccionar vista", tabs)

if selected_tab == "Mapa de viviendas":
    # Mostrar el mapa con los puntos filtrados si existe anuncios_filtrados_con_prediccion
    if anuncios_filtrados_con_prediccion is not None and 'lat' in anuncios_filtrados_con_prediccion.columns and 'lon' in anuncios_filtrados_con_prediccion.columns:
        # Crear un mapa folium
        mymap = folium.Map(location=[40.4165, -3.70256], zoom_start=12)

        # Iterar sobre las filas del DataFrame y añadir marcadores al mapa
        for index, row in anuncios_filtrados_con_prediccion.iterrows():
            color = row['color']
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=5,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=f"<b>Recomendación de inversión:</b> {row['RECOMENDACION_INVERSION']}<br>"
                      f"<b>Precio predicción:</b> {row['PREDICCION_PRECIO']:.2f}€<br>"
                      f"<b>Precio actual:</b> {row['FINALPRICE_DISCOUNT']:.2f}€"
            ).add_to(mymap)

        # Mostrar el mapa en Streamlit
        folium_static(mymap)

elif selected_tab == "Viviendas filtradas":
    # Mostrar las filas filtradas del dataset con una columna adicional "¿Inversión recomendable?"
    if anuncios_filtrados_con_prediccion is not None:
        # Ordenar por RECOMENDACION_INVERSION y diferencia porcentual
        anuncios_filtrados_con_prediccion['DIFERENCIA_PORCENTUAL'] = (anuncios_filtrados_con_prediccion['PREDICCION_PRECIO'] - anuncios_filtrados_con_prediccion['FINALPRICE_DISCOUNT']) / anuncios_filtrados_con_prediccion['FINALPRICE_DISCOUNT']
        anuncios_filtrados_con_prediccion.sort_values(by=['RECOMENDACION_INVERSION', 'DIFERENCIA_PORCENTUAL'], ascending=[False, False], inplace=True)

        # Renombrar columnas
        anuncios_filtrados_con_prediccion.rename(columns=column_rename_dict, inplace=True)

        # Definir orden de columnas para mostrar
        columns_to_display = ['Recomendación de Inversión', 'Precio Actual', 'Precio Predicho', 'Número de Habitaciones',
                              'Número de Baños', 'Año de Construcción', 'Metros Cuadrados', 'Distancia al Metro',
                              'Distancia a la Castellana', 'Distancia al Centro']

        st.dataframe(anuncios_filtrados_con_prediccion[columns_to_display].reset_index(drop=True))

