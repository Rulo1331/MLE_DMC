import streamlit as st
import requests

# Configuración de página
st.set_page_config(page_title="API de Aprobacion de Credito con Riesgo Politico y Perfil Digital", layout="centered")
st.title("📈 API de Aprobacion de Credito con Riesgo Politico y Perfil Digital")
st.markdown("predecir si el cliente está aprobado o no.")

# Inputs del cliente
age = st.slider("🎂 Edad", 18, 80, 40)
income = st.number_input("💵 Ingreso Mensual (USD)", min_value=0.0, step=100.0, value=2000.0)
app = st.slider("📆 Uso de App ", 0, 10, 5)
digital = st.slider("📄 Fuerza de Perfil Digital", 5, 100, 1)
contacts = st.slider("📄 Numero de Contactos", 21, 65, 1)
residence = st.selectbox("💼 Riesgo de Zona", ["media", "baja", "alta"])
political = st.radio("📬 Evento Politico Ultimo Mes", ["No", "Sí"])

# Threshold slider
threshold = st.slider("🎚 Umbral de aceptación (threshold)", 0.0, 1.0, 0.5, step=0.01)

# Botón de predicción
if st.button("🔍 Evaluar Probabilidad"):
    with st.spinner("Consultando modelo..."):
        try:
            payload = {
                "age": age,
                "monthly_income_usd": income,
                "app_usage_score": app,
                "digital_profile_strength": digital,
                "num_contacts_uploaded": contacts,
                "residence_risk_zone": residence,
                "political_event_last_month": 1 if political == "Sí" else 0,
                "threshold": threshold
            }

            r = requests.post("http://localhost:8000//predict_aprobacion", json=payload)
            if r.status_code == 200:
                resultado = r.json()
                score = resultado["score_probabilidad"]
                aceptara = resultado["aceptará"]

                st.markdown(f"### 🔢 Score de aceptación: **{score:.3f}**")
                st.markdown(f"### 🎯 Umbral usado: **{threshold:.2f}**")

                if aceptara:
                    st.success("✅ El cliente probablemente **será aprobado** .")
                else:
                    st.warning("⚠️ El cliente probablemente **será rechazado** .")
            else:
                st.error("❌ Error en la respuesta del modelo.")
        except Exception as e:
            st.error(f"❌ No se pudo conectar al API: {e}")


#uvicorn api:app --reload --port 8000
#streamlit run app.py
