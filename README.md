# face-recognition-project 
Ce projet permet de détecter les visages, d’estimer l’âge et le genre à partir d’une image ou d’une webcam en utilisant OpenCV et des modèles Caffe pré-entraînés.

## Structure du projet
```bash
├── 📁 models
│   ├── 📄 age_deploy.prototxt
│   ├── 📄 age_net.caffemodel
│   ├── 📄 deploy.prototxt
│   ├── 📄 gender_deploy.prototxt
│   ├── 📄 gender_net.caffemodel
│   └── 📄 res10_300x300_ssd_iter_140000.caffemodel
├── 📝 README.md
├── 🐍 age_detection.py
├── 🐍 face_detection.py
├── 🐍 gender_detector.py
├── 🐍 main.py
└── 📄 requirement.txt
```
## Fonctionnement de chaque fichier

### face_detection.py
- Détecte les visages dans une image ou une vidéo en utilisant le modèle `res10_300x300_ssd_iter_140000.caffemodel`.
- Retourne les coordonnées des visages pour que les autres modules puissent les utiliser.

### age_detection.py
- Utilise `age_net.caffemodel` et `age_deploy.prototxt` pour estimer l’âge de chaque visage détecté.
- Renvoie l’âge estimé sous forme de texte (ex. : "25-32 ans").

### gender_detector.py
- Utilise `gender_net.caffemodel` et `gender_deploy.prototxt` pour déterminer le genre de chaque visage détecté.
- Renvoie le genre sous forme de texte ("Homme" ou "Femme").

### main.py
- Script principal du projet.
- Lit la webcam ou des images.
- Passe chaque image à `FaceDetector` pour détecter les visages.
- Pour chaque visage détecté, utilise `AgeDetector` et `GenderDetector` pour afficher l’âge et le genre.
- Affiche le résultat en temps réel avec OpenCV : rectangle autour du visage + texte âge/genre.

---
### Installer les dépendances
```bash
pip install -r requirement.txt
```
### Lancer le projet 
```bash
python main.py
```
