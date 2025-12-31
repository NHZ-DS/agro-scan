import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# --- CONFIGURATION DU DASHBOARD ---
st.set_page_config(
    page_title="Agro-Scan AI",
    page_icon="🌻",
    layout="centered"
)

# --- CSS PERSONNALISÉ (UI/UX) ---
st.markdown("""
    <style>
    .stApp { background-color: #F1F8E9; }
    h1 { color: #33691E; text-align: center; font-family: 'Helvetica Neue', sans-serif; }
    .stButton>button { background-color: #689F38; color: white; border-radius: 20px; width: 100%; border: none; padding: 10px; font-weight: bold; }
    .result-box { padding: 20px; background-color: white; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; margin-top: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER ---
st.title("🌻 Agro-Scan")
st.markdown("<p style='text-align: center;'>Votre assistant expert en botanique</p>", unsafe_allow_html=True)

# --- GESTION DU MODÈLE ---
@st.cache_resource
def load_model():
    """
    Charge le modèle TensorFlow (.h5) en cache pour éviter de le recharger
    à chaque interaction utilisateur (Optimisation Latence).
    """
    model = tf.keras.models.load_model('agro_scan_model.h5')
    return model

with st.spinner('Initialisation du moteur d\'inférence...'):
    model = load_model()

# --- CONSTANTES ---
# L'ordre des classes doit correspondre exactement aux indices de l'entraînement (train_generator.class_indices)
CLASS_NAMES = ['Marguerite (Daisy)', 'Pissenlit (Dandelion)', 'Rose', 'Tournesol (Sunflower)', 'Tulipe']

# Base de connaissances métier (Agri-Tech)
ADVICE = {
    'Marguerite (Daisy)': "🌸 **Conseil :** Idéale pour les bordures. Aime le plein soleil.",
    'Pissenlit (Dandelion)': "🥗 **Info :** Comestible ! Les feuilles se mangent en salade.",
    'Rose': "🌹 **Soins :** Attention aux pucerons. Arrosez au pied, pas sur les feuilles.",
    'Tournesol (Sunflower)': "☀️ **Culture :** Pivote vers le soleil. A besoin de beaucoup d'eau.",
    'Tulipe': "🌷 **Plantation :** Plantez les bulbes en automne avant les gelées."
}

# --- PIPELINE DE PRÉDICTION ---
def predict(image_data, model):
    """
    Prépare l'image et effectue l'inférence via le modèle CNN.
    
    Étapes :
    1. Convertir en RGB
    2. Resize : 160x160 (Contrainte d'entrée MobileNetV2).
    3. Normalisation : Pixel / 255.0 (Mise à l'échelle 0-1 comme lors de l'entraînement).
    4. Batching : Ajout d'une dimension pour créer un tenseur (1, 160, 160, 3).
    """
    # 1. Conversion en RGB (important de faire ça avant le redimensionnement)
    image_data = image_data.convert('RGB')

    # 2. Redimensionnement avec filtre LANCZOS pour préserver la qualité des détails
    size = (160, 160)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)

    # 3. Conversion en tableau NumPy
    img_array = np.asarray(image)
    
    # 4. Normalisation (Scaling)
    normalized_image_array = (img_array.astype(np.float32) / 255.0)
    
    # 5. Expansion de dimension (Batch Dimension)
    data = np.expand_dims(normalized_image_array, axis=0)

    # 6. Inférence
    prediction = model.predict(data)
    return prediction


# --- INTERFACE UTILISATEUR ---
uploaded_file = st.file_uploader("📸 Importez une image pour analyse", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Affichage responsive
    st.image(image, caption='Image source', use_container_width=True)
    
    if st.button("🔍 Lancer le Diagnostic"):
        with st.spinner("Analyse biométrique en cours..."):
            predictions = predict(image, model)
            
            # Post-traitement des probabilités
            class_index = np.argmax(predictions[0])
            result_text = CLASS_NAMES[class_index]
            confidence = np.max(predictions[0]) * 100
            
            # Affichage du résultat
            st.markdown(f"""
            <div class="result-box">
                <h2>Identification : <span style="color:#33691E">{result_text}</span></h2>
                <p>Indice de confiance : <b>{confidence:.2f}%</b></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Affichage du conseil agronomique
            st.info(ADVICE.get(result_text, "Analyse terminée."))
