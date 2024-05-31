import cv2
import numpy as np
import os

name = "Nic Cage"
picture = "Nic Cage.jpg"

f_list = []

haarcascade_path = "haarcascade_frontalface_default.xml"
classifier = cv2.CascadeClassifier(haarcascade_path)

if classifier.empty():
    print(f"Error loading {haarcascade_path}")
else:
    print(f"Successfully loaded {haarcascade_path}")

# Ensure the 'picture' is a directory or adjust accordingly
if os.path.isdir(picture):
    for filename in os.listdir(picture):
        picture_path = os.path.join(picture, filename)
        frame = cv2.imread(picture_path)
        if frame is None:
            print(f"Error reading picture {picture_path}")
            continue

        gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = classifier.detectMultiScale(gray_scale, 1.5, 5)

        if len(faces) == 0:
            print(f"No face found in {picture_path}")
        else:
            print(f"Face found in {picture_path}: {len(faces)}")

        for (x, y, w, h) in faces:
            im_face = frame[y:y + h, x:x + w]
            gray_face = cv2.cvtColor(im_face, cv2.COLOR_BGR2GRAY)
            gray_face = cv2.resize(gray_face, (100, 100))
            f_list.append(gray_face.reshape(-1))
            if len(f_list) == 10:
                break
        if len(f_list) == 10:
            break
else:
    frame = cv2.imread(picture)
    if frame is None:
        print(f"Error reading picture {picture}")
    else:
        gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = classifier.detectMultiScale(gray_scale, 1.5, 5)

        if len(faces) == 0:
            print(f"No face found in {picture}")
        else:
            print(f"Face found in {picture}: {len(faces)}")

        for (x, y, w, h) in faces:
            im_face = frame[y:y + h, x:x + w]
            gray_face = cv2.cvtColor(im_face, cv2.COLOR_BGR2GRAY)
            gray_face = cv2.resize(gray_face, (100, 100))
            f_list.append(gray_face.reshape(-1))
            if len(f_list) == 10:
                break

if len(f_list) == 0:
    print("No faces detected to write to CSV")
else:
    output_filename = f"{name.replace(' ', '_')}_faces.csv"
    np.savetxt(output_filename, np.array(f_list), delimiter=",")
    print(f"Face data written to {output_filename}")