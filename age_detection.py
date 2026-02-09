import cv2 as cv

class AgeDetector:
    def __init__(self):
        # Chemins des modèles et des fichiers de configuration
        self.modelFile = "models/age_net.caffemodel"
        self.configFile = "models/age_deploy.prototxt"
        self.net = cv.dnn.readNetFromCaffe(self.configFile, self.modelFile)

        self.ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
                        '(25-32)', '(38-43)', '(48-53)', '(60-100)']

    def predict(self, face_img):
        blob = cv.dnn.blobFromImage(face_img, 1.0, (227, 227),
                                     (78.4263377603, 87.7689143744, 114.895847746),
                                     swapRB=False)
        self.net.setInput(blob)
        preds = self.net.forward()
        return self.ageList[preds[0].argmax()]
