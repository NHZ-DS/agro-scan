# 🌿 Agro-Scan : Plant Disease Detection AI

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![MobileNetV2](https://img.shields.io/badge/Model-MobileNetV2-green?style=for-the-badge)

Une application de Computer Vision capable d'identifier les espèces végétales et de diagnostiquer leur état de santé à partir d'une simple photo.
Ce projet utilise le **Transfer Learning** pour offrir une haute précision avec une architecture légère adaptée au mobile.

## 🏗️ Architecture Technique

Le cœur du système repose sur une architecture **MobileNetV2** pré-entraînée sur ImageNet, affinée (Fine-Tuning) sur un dataset spécifique.

* **Base Model :** MobileNetV2 (Choisi pour sa faible latence et sa taille < 20Mo, idéal pour l'Edge Computing).
* **Custom Head :** GlobalAveragePooling2D + Dense Layers (128 neurones) + Dropout (0.2) pour éviter l'overfitting.
* **Training :** Entraînement réalisé sur GPU (Google Colab T4) avec Data Augmentation (Rotation, Zoom, Shift).

## ✨ Fonctionnalités

* 📸 **Reconnaissance d'images :** Upload de fichiers JPG/PNG.
* 🧠 **Inférence en temps réel :** Analyse rapide (< 500ms).
* 💡 **Smart Advice :** Le système ne donne pas juste le nom de la plante, mais fournit des conseils agronomiques adaptés (Arrosage, Exposition, Soins).

## 🛠️ Installation & Démarrage

1.  **Cloner le projet**
    ```bash
    git clone [https://github.com/NHZ-DS/agro-scan.git](https://github.com/NHZ-DS/agro-scan.git)
    cd agro-scan
    ```

2.  **Créer l'environnement virtuel**
    ```bash
    python -m venv venv
    # Windows :
    .\venv\Scripts\activate
    # Mac/Linux :
    source venv/bin/activate
    ```

3.  **Installer les dépendances**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Lancer l'application**
    ```bash
    streamlit run app.py
    ```

## 📊 Performance & Choix du Modèle

Pourquoi **MobileNetV2** plutôt que ResNet50 ?
Dans un contexte agricole ("Agro-Tech"), la connectivité est souvent limitée. J'ai privilégié une architecture optimisée pour le mobile (**Depthwise Separable Convolutions**) qui permet une inférence rapide même sur des appareils à faible puissance de calcul, contrairement à des modèles plus lourds comme VGG16.

## 👤 Auteur
**Nouhaila ERRAMI** - Étudiante en Mathématiques et Data Science
