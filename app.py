import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Agro-Scan AI",
    page_icon="🌻",
    layout="centered"
)

# --- STYLE CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #F1F8E9; }
    h1 { color: #33691E; text-align: center; font-family: 'Helvetica Neue', sans-serif; }
    .stButton>button { background-color: #689F38; color: white; border-radius: 20px; width: 100%; border: none; padding: 10px; font-weight: bold; }
    .result-box { padding: 20px; background-color: white; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; margin-top: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- TITRE ---
st.title("🌻 Agro-Scan")
st.markdown("<p style='text-align: center;'>Votre assistant expert en botanique</p>", unsafe_allow_html=True)

# --- CHARGEMENT DU MODÈLE ---
@st.cache_resource
def load_model():
    # Assure-toi que le fichier .h5 est bien dans le même dossier
    model = tf.keras.models.load_model('agro_scan_model.h5')
    return model

with st.spinner('Démarrage du moteur IA...'):
    model = load_model()

# --- CLASSES ---
CLASS_NAMES = ['Marguerite (Daisy)', 'Pissenlit (Dandelion)', 'Rose', 'Tournesol (Sunflower)', 'Tulipe']

# --- CONSEILS ---
ADVICE = {
    'Marguerite (Daisy)': "🌸 **Conseil :** Idéale pour les bordures. Aime le plein soleil.",
    'Pissenlit (Dandelion)': "🥗 **Info :** Comestible ! Les feuilles se mangent en salade.",
    'Rose': "🌹 **Soins :** Attention aux pucerons. Arrosez au pied, pas sur les feuilles.",
    'Tournesol (Sunflower)': "☀️ **Culture :** Pivote vers le soleil. A besoin de beaucoup d'eau.",
    'Tulipe': "🌷 **Plantation :** Plantez les bulbes en automne avant les gelées."
}

# --- PRÉDICTION ---
def predict(image_data, model):
    size = (160, 160)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image)
    normalized_image_array = (img_array.astype(np.float32) / 255.0)
    data = np.expand_dims(normalized_image_array, axis=0)
    prediction = model.predict(data)
    return prediction

# --- INTERFACE ---
uploaded_file = st.file_uploader("📸 Choisissez une photo de fleur", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # CORRECTION ICI : use_container_width=True remplace use_column_width
    st.image(image, caption='Photo analysée', use_container_width=True)
    
    if st.button("🔍 Identifier la plante"):
        with st.spinner("Analyse des pétales en cours..."):
            predictions = predict(image, model)
            class_index = np.argmax(predictions[0])
            result_text = CLASS_NAMES[class_index]
            confidence = np.max(predictions[0]) * 100
            
            st.markdown(f"""
            <div class="result-box">
                <h2>C'est une <span style="color:#33691E">{result_text}</span> !</h2>
                <p>Certitude IA : {confidence:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.info(ADVICE.get(result_text, "Pas de conseil spécifique."))