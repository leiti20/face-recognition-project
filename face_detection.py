import cv2 as cv 
 
class FaceDetector:
    def __init__(self):
        # Chemins des modèles et des fichiers de configuration
        self.modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
        self.configFile = "models/deploy.prototxt"
        self.net = cv.dnn.readNetFromCaffe(self.configFile, self.modelFile)

    def detect_faces(self, frame):
        # Obtenir les dimensions de l'image
        h, w = frame.shape[:2]
        
        # Prétraiter l'image pour le réseau de neurones
        blob = cv.dnn.blobFromImage(frame, 1.0, (300, 300),(104, 177, 123))

        # Passer l'image à travers le réseau pour obtenir les détections
        self.net.setInput(blob)
        detections = self.net.forward()

        faces = []
        # Parcourir les détections et filtrer celles qui ont une confiance suffisante
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.6:
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                x1, y1, x2, y2 = box.astype(int)
                faces.append((x1, y1, x2, y2))

        return faces
