import cv2 as cv

class GenderDetector:
    def __init__(self):
        # Chemins des modèles et des fichiers de configuration
        self.modelFile = "models/gender_net.caffemodel"
        self.configFile = "models/gender_deploy.prototxt"
        self.net = cv.dnn.readNetFromCaffe(self.configFile, self.modelFile)

        self.genderList = ['Homme', 'Femme']

    def predict(self, face_img):
        blob = cv.dnn.blobFromImage(face_img, 1.0, (227, 227),
                                     (78.4263377603, 87.7689143744, 114.895847746),
                                     swapRB=False)
        self.net.setInput(blob)
        preds = self.net.forward()
        return self.genderList[preds[0].argmax()]
