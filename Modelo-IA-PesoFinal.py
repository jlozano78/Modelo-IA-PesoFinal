import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy.optimize import minimize
import streamlit as st

# Convertir variables categóricas a valores numéricos
from sklearn.preprocessing import LabelEncoder


#Configuracion de streamlit
st.title('Modelo de IA para identificar lasprincipales variables que influyen en el peso final')
st.write('''
        Esta aplicacion permite realizar las predicciones del peso final del pollo incluyendo el consumo de alimentos en la fase 'Acabado -Finalizador' asi como tamboien con la estacion del corral (Veran, Otoño, Invierno,Primavera) y finalizando con el sexo del pollo en mencion tabien se puede visualizar la comparacion de peso real vs el peso predicho por el algoritmo''')

#CARGAR DATOS
uploaded_file = st.file_uploader("Cargar Archivo CSV" ,type=['csv'])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

        # Excluir la columna IEP de la selección inicial
        X = df.drop(['PesoFinal', 'SiNoPesoFinal', 'ICA', 'MortStd', 'IEP', 'GananciaDiaVenta', 'DiasSaca', 'EdadGranja',
            'ConsumoFinalizador', 'PesoStd', 'PobInicial', 'Nzona', 'PorMortFinal', 'StdConsAve'], axis=1)  # Todas las variables menos PesoFinal, SiNoPesoFinal e IEP
        y = df['SiNoPesoFinal']  # Variable objetivo (Si/No)

        # Seleccionar las 20 variables más importantes (excluyendo IEP)
        selector = SelectKBest(f_classif, k=20)
        selector.fit(X, y)

        # Obtener las variables seleccionadas y asegurarse de que IEP no esté incluido
        selected_features = X.columns[selector.get_support()]

        # Si IEP está dentro de las 20 principales, añadir la siguiente mejor característica
        k = 20
        while 'IEP' in selected_features:
            k += 1
            selector = SelectKBest(f_classif, k=k)
            selector.fit(X, y)
            selected_features = X.columns[selector.get_support()]

        # Imprimir las variables seleccionadas
        st.write("Características seleccionadas:", selected_features)

        # Crear y entrenar un modelo de aprendizaje
        X_train, X_test, y_train, y_test = train_test_split(X[selected_features], y, test_size=0.2, random_state=42)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        # Evaluar el modelo
        y_pred = model.predict(X_test)
        st.write("Accuracy:", accuracy_score(y_test, y_pred))
        st.write(classification_report(y_test, y_pred))

        # Importancia de las características
        importances = model.feature_importances_
        feature_importances = pd.DataFrame({'Característica': selected_features, 'Importancia': importances * 100})
        feature_importances['Importancia'] = feature_importances['Importancia'].round(2)
        feature_importances = feature_importances.sort_values('Importancia', ascending=False)

        # Sugerencias para mejorar el objetivo "SI"
        st.write("\nPara mejorar el logro del objetivo 'SI' en PesoFinal, concéntrate en mejorar las siguientes variables:")
        for feature, importance in feature_importances.itertuples(index=False):
            st.write(f"- {feature} (Importancia: {importance:.2f}%)")

    except Exception as e:
            st.error(f"Error al leer el archivo Csv: {e}")