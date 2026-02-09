import cv2 as cv
from face_detection import FaceDetector
from gender_detector import GenderDetector
from age_detection import AgeDetector

face_detector = FaceDetector()
gender_detector = GenderDetector()
age_detector = AgeDetector()

cap = cv.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = face_detector.detect_faces(frame)
    for (x1, y1, x2, y2) in faces:
        face_img = frame[y1:y2, x1:x2]

        if face_img.size == 0:
            continue

        gender = gender_detector.predict(face_img)
        age = age_detector.predict(face_img)

        label = f"{gender}, {age}"

        cv.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv.putText(frame, label, (x1, y1-10),
                    cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv.imshow("Age & Gender Detection", frame)
    # appuyer sur 'q' pour quitter
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
