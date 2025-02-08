import streamlit as st
import joblib  # Pour charger le modèle
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Charger le modèle entraîné
model = joblib.load("weather_model3.pkl")  
# Titre de l'application
st.title("🌤️ Application de Prédiction DE LA TEMPERATURE")

# Interface utilisateur pour saisir les données
st.sidebar.header("Entrez les paramètres météo")

# Entrées pour chaque variable du modèle
clouds = st.sidebar.slider("Couverture nuageuse (%)", min_value=0, max_value=100, value=50)
clouds_low = st.sidebar.slider("Couverture nuageuse basse (%)", min_value=0, max_value=100, value=50)
wind_dir = st.sidebar.number_input("Direction du vent (°)", min_value=0, max_value=360, value=90)
ts = st.sidebar.number_input("Timestamp", min_value=0, value=0)
ozone = st.sidebar.number_input("Niveau d'ozone (DU)", min_value=0.0)
wind_spd = st.sidebar.number_input("Vitesse du vent (m/s)", min_value=0.0)
wind_gust_spd = st.sidebar.number_input("Rafales de vent (m/s)", min_value=0.0)
rh = st.sidebar.slider("Humidité relative (%)", min_value=0, max_value=100, value=50)
dewpt = st.sidebar.number_input("Point de rosée (°C)", min_value=-50.0, max_value=50.0, value=15.0)
app_temp = st.sidebar.number_input("Température ressentie (°C)", min_value=-50.0, max_value=50.0, value=20.0)
ghi = st.sidebar.number_input("Global Horizontal Irradiance (W/m²)", min_value=0.0)
wind_cdir = st.sidebar.selectbox("Wind Cardinal Direction", options=["N", "NE", "E", "SE", "S", "SW", "W", "NW"])

# Bouton pour faire la prédiction
if st.sidebar.button("Prédire la température"):
    # Initialiser le LabelEncoder pour la direction du vent
     encoder = LabelEncoder()
     wind_directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
     encoder.fit(wind_directions)

         # Encoder la direction du vent sélectionnée
     wind_cdir_encoded = encoder.transform([wind_cdir])[0]
    # Transformer les données en format utilisable par le modèle
     features = np.array([[clouds, clouds_low, wind_dir, ts, ozone, wind_spd, 
                          wind_gust_spd,rh,dewpt, app_temp,ghi,wind_cdir_encoded]])

    # Prédiction du modèle
     prediction = model.predict(features)

    # Affichage du résultat
     st.subheader("🌦️ Prévision de la température :")
     temperature = prediction[0]  # Récupérer uniquement la température, sans probabilité de pluie

     st.write(f"🌡️ Température prévue : **{temperature:.2f}°C**")
