
import cv2
import numpy as np
import pandas as pd

class KNN:
    def __init__(self, K=3):
        self.K = K
    
    def fit(self, x_train, y_train):
        self.X_train = x_train
        self.Y_train = y_train
    
    def predict(self, X_test):
        predictions = []
        for x_test in X_test:
            distances = np.linalg.norm(self.X_train - x_test, axis=1)
            k_indices = distances.argsort()[:self.K]
            k_labels = self.Y_train[k_indices]
            predictions.append(np.bincount(k_labels).argmax())
        return predictions

data = pd.read_csv("Nic_Cage_faces.csv").values
X, Y = data[:, :-1], data[:, -1]

model = KNN(K=5)
model.fit(X, Y)

cap = cv2.VideoCapture(0)
classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray, 1.5, 5)
    X_test = []

    for (x, y, w, h) in faces:
        im_face = gray[y:y + h, x:x + w]
        im_face = cv2.resize(im_face, (100, 100)).reshape(1, -1)
        X_test.append(im_face)

    if len(faces) > 0:
        response = model.predict(np.array(X_test))
        for i, (x, y, w, h) in enumerate(faces):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
            cv2.putText(frame, response[i], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()