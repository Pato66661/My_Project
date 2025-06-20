import streamlit as st
import joblib
import numpy as np

# Cargar el modelo
model = joblib.load("modelo_random.pkl")

# Configurar la página
st.set_page_config(page_title="Predictor Cardíaco", layout="wide")

# Título y descripción
st.title("Sistema de Predicción de Enfermedades Cardíacas")
st.markdown("""
Complete el formulario con los datos del paciente para obtener una predicción.
""")

# Crear el formulario en la barra lateral
with st.sidebar:
    st.header("Datos del Paciente")
    
    age = st.slider("Edad", 20, 100, 50)
    sex = st.radio("Sexo", options=[("Hombre", 1), ("Mujer", 0)], format_func=lambda x: x[0])[1]
    cp = st.selectbox("Tipo de dolor torácico", options=[0, 1, 2, 3], 
                     format_func=lambda x: ["Típico", "Atípico", "No anginoso", "Asintomático"][x])
    
    # Más campos del formulario
    trestbps = st.number_input("Presión arterial en reposo (mm Hg)", 80, 200, 120)
    chol = st.number_input("Colesterol sérico (mg/dl)", 100, 600, 200)
    fbs = st.radio("Azúcar en ayunas > 120 mg/dl", options=[("Sí", 1), ("No", 0)], format_func=lambda x: x[0])[1]
    
    # Continuar con el resto de campos...
    restecg = st.selectbox("Resultados electrocardiográficos en reposo", options=[0, 1, 2], 
                          format_func=lambda x: ["Normal", "Anormalidad ST-T", "Hipertrofia ventricular"][x])
    thalach = st.number_input("Frecuencia cardíaca máxima alcanzada", 60, 220, 150)
    exang = st.radio("Angina inducida por ejercicio", options=[("Sí", 1), ("No", 0)], format_func=lambda x: x[0])[1]
    oldpeak = st.slider("Depresión del ST inducida por ejercicio", 0.0, 6.2, 1.0)
    slope = st.selectbox("Pendiente del segmento ST de ejercicio máximo", options=[0, 1, 2], 
                        format_func=lambda x: ["Ascendente", "Plano", "Descendente"][x])
    ca = st.slider("Número de vasos principales coloreados por fluoroscopia", 0, 3, 0)
    thal = st.selectbox("Resultado de la talasemia", options=[0, 1, 2, 3], 
                       format_func=lambda x: ["Normal", "Defecto fijo", "Defecto reversible", "Otro"][x])

# Botón para realizar la predicción
if st.button("Predecir Riesgo Cardíaco"):
    try:
        # Preparar los datos para el modelo
        features = np.array([[
            age, sex, cp, trestbps, chol, fbs, restecg,
            thalach, exang, oldpeak, slope, ca, thal
        ]])
        
        # Realizar predicción
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0][1] * 100
        
        # Mostrar resultados
        st.subheader("Resultado de la Predicción")
        
        if prediction == 1:
            st.error(f"⚠️ Riesgo de enfermedad cardíaca detectado (Probabilidad: {proba:.2f}%)")
            st.markdown("""
            **Recomendaciones:**
            - Consulte a un cardiólogo lo antes posible
            - Realice cambios en su estilo de vida
            - Controle sus factores de riesgo regularmente
            """)
        else:
            st.success(f"✅ No se detectó riesgo cardíaco significativo (Probabilidad: {proba:.2f}%)")
            st.markdown("""
            **Recomendaciones:**
            - Mantenga hábitos de vida saludables
            - Realice chequeos periódicos
            - Continúe con la prevención
            """)
            
        # Mostrar barra de probabilidad
        st.progress(int(proba))
        st.caption(f"Probabilidad de enfermedad cardíaca: {proba:.2f}%")
        
    except Exception as e:
        st.error(f"Error al procesar la solicitud: {str(e)}")

# Información adicional en el pie de página
st.markdown("---")
st.caption("""
**Nota:** Este predictor utiliza un modelo de aprendizaje automático y no sustituye el diagnóstico médico profesional. 
Consulte siempre a un especialista para evaluación clínica.
""")
