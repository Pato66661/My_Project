import streamlit as st
import joblib
import numpy as np
from PIL import Image  # Para manejar imágenes

# Cargar el modelo
@st.cache_resource
def load_model():
    return joblib.load("modelo_random.pkl")

model = load_model()

# Configuración de la página
st.set_page_config(
    page_title="Predictor Cardíaco Avanzado",
    #page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
    }
    .stButton>button:hover {
        background-color: #ff2b2b;
        color: white;
    }
    .result-box {
        padding: 2rem;
        border-radius: 1rem;
        margin: 1rem 0;
    }
    .high-risk {
        background-color: #ffebee;
        border-left: 0.5rem solid #ff1744;
    }
    .low-risk {
        background-color: #e8f5e9;
        border-left: 0.5rem solid #00c853;
    }
</style>
""", unsafe_allow_html=True)

# Encabezado con imagen
col1, col2 = st.columns([1, 3])
with col1:
    st.image("heart_icon.png", width=100)  # Asegúrate de tener esta imagen o usa una URL
with col2:
    st.title("Predictor Avanzado de Riesgo Cardíaco")
    st.caption("Complete el formulario para evaluar el riesgo de enfermedad cardiovascular")

# Formulario en un contenedor expandible
with st.expander("📋 Formulario del Paciente", expanded=True):
    with st.form("patient_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.slider("Edad", 20, 100, 50, help="Edad del paciente en años")
            sex = st.radio("Sexo", options=[("Mujer", 0), ("Hombre", 1)], format_func=lambda x: x[0], help="Sexo biológico del paciente")[1]
            cp = st.selectbox("Tipo de dolor torácico", options=[0, 1, 2, 3], 
                            format_func=lambda x: ["Típico", "Atípico", "No anginoso", "Asintomático"][x],
                            help="Tipo de dolor en el pecho reportado")
            trestbps = st.number_input("Presión arterial (mm Hg)", 80, 200, 120, 
                                     help="Presión arterial en reposo en mm Hg")
            
        with col2:
            chol = st.number_input("Colesterol (mg/dl)", 100, 600, 200,
                                 help="Colesterol sérico en mg/dl")
            fbs = st.radio("Glucosa > 120 mg/dl", options=[("No", 0), ("Sí", 1)], 
                          format_func=lambda x: x[0], 
                          help="Azúcar en sangre en ayunas > 120 mg/dl")[1]
            restecg = st.selectbox("Electrocardiograma", options=[0, 1, 2], 
                                 format_func=lambda x: ["Normal", "Anormalidad ST-T", "Hipertrofia ventricular"][x],
                                 help="Resultados electrocardiográficos en reposo")
            thalach = st.number_input("Frecuencia cardíaca máxima", 60, 220, 150,
                                    help="Frecuencia cardíaca máxima alcanzada")
            
        with col3:
            exang = st.radio("Angina por ejercicio", options=[("No", 0), ("Sí", 1)], 
                            format_func=lambda x: x[0],
                            help="Angina inducida por ejercicio")[1]
            oldpeak = st.slider("Depresión del ST", 0.0, 6.2, 1.0, step=0.1,
                              help="Depresión del ST inducida por ejercicio relativo al descanso")
            slope = st.selectbox("Pendiente del ST", options=[0, 1, 2], 
                               format_func=lambda x: ["Ascendente", "Plano", "Descendente"][x],
                               help="Pendiente del segmento ST de ejercicio máximo")
            ca = st.slider("Vasos principales", 0, 3, 0,
                          help="Número de vasos principales coloreados por fluoroscopia")
            thal = st.selectbox("Talasemia", options=[0, 1, 2, 3], 
                              format_func=lambda x: ["Normal", "Defecto fijo", "Defecto reversible", "Otro"][x],
                              help="Resultado de la prueba de talasemia")
        
        # Botones en una fila
        col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 2])
        
        with col_btn2:
            submitted = st.form_submit_button("🔍 Analizar Riesgo", use_container_width=True)
        
        with col_btn1:
            if st.form_submit_button("🧹 Limpiar Formulario", type="secondary", use_container_width=True):
                st.session_state.clear()
                st.experimental_rerun()

# Procesamiento de resultados
if submitted:
    try:
        features = np.array([[
            age, sex, cp, trestbps, chol, fbs, restecg,
            thalach, exang, oldpeak, slope, ca, thal
        ]])
        
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0][1] * 100
        
        # Mostrar resultados
        st.markdown("---")
        st.subheader("Resultados del Análisis")
        
        risk_class = "high-risk" if prediction == 1 else "low-risk"
        #result_icon = "⚠️" if prediction == 1 else "✅"
        result_text = "Riesgo Elevado" if prediction == 1 else "Riesgo Bajo"
        
        st.markdown(f"""
        <div class="result-box {risk_class}">
            <h3>{result_icon} {result_text} - Probabilidad: {proba:.1f}%</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Barra de probabilidad
        st.progress(int(proba), text=f"Probabilidad de enfermedad cardíaca: {proba:.1f}%")
        
        # Recomendaciones específicas
        if prediction == 1:
            st.warning("""
            **Recomendaciones Médicas:**
            - Consulta urgente con cardiólogo
            - Realizar electrocardiograma y análisis de sangre
            - Monitorear presión arterial regularmente
            - Implementar cambios en dieta y ejercicio
            - Considerar evaluación de estrés cardíaco
            """)
        else:
            st.success("""
            **Recomendaciones Preventivas:**
            - Mantener chequeos anuales
            - Dieta balanceada y ejercicio regular
            - Controlar niveles de colesterol y presión
            - Evitar tabaco y consumo excesivo de alcohol
            - Manejar niveles de estrés
            """)
        
        # Gráfico de factores de riesgo (simplificado)
        factors = {
            'Edad': age/100,
            'Presión Arterial': trestbps/200,
            'Colesterol': chol/600,
            'Depresión ST': oldpeak/6.2
        }
        
        st.bar_chart(factors)
        st.caption("Factores de riesgo principales (valores normalizados)")
        
    except Exception as e:
        st.error(f"❌ Error en el análisis: {str(e)}")
        st.info("Por favor verifique los datos ingresados e intente nuevamente")

# Pie de página informativo
st.markdown("---")
st.markdown("""
**Nota importante:**  
Este sistema predictivo utiliza inteligencia artificial para estimar riesgos basados en datos clínicos, 
pero **no sustituye el criterio médico profesional**. Siempre consulte con un especialista en cardiología 
para diagnóstico y tratamiento.
""")

# Mostrar datos técnicos en expander
with st.expander("ℹ Información Técnica"):
    st.markdown("""
    **Modelo utilizado:** Random Forest Classifier  
    **Precisión del modelo:** ~85% (validación cruzada)  
    **Variables consideradas:** 13 factores clínicos  
    **Desarrollado por:** [Tu Nombre o Institución]  
    
    *Última actualización: Enero 2024*
    """)
