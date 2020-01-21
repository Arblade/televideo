import cv2
import operator
import pickle
from numpy import *
from imutils.object_detection import non_max_suppression
import time


def reco_vd():
    label = []
    faces_all = []
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flip_frame = cv2.flip(gray, 1)
    faces = array(face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4, minSize=(50, 50)))
    profile = array(profile_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4, minSize=(50, 50)))
    profile_flip = array(
        profile_cascade.detectMultiScale(flip_frame, scaleFactor=1.4, minNeighbors=4, minSize=(50, 50)))
    for (x, y, w, h) in faces:
        faces_all.append([x, y, x + w, y + h])
    for (x, y, w, h) in profile_flip:
        faces_all.append([width - (x + w), y, width - x, y + h])
    for (x, y, w, h) in profile:
        faces_all.append([x, y, x + w, y + h])
    faces_all = array(faces_all)
    rect = non_max_suppression(faces_all, probs=None, overlapThresh=0.65)
    for (x, y, w, h) in rect:
        roi_gray = gray[y: h, x: w]
        id_, conf = reco.predict(roi_gray)
        if conf <= 80:
            name = labels[id_]
        else:
            name = "Inconnu"
        label.append(name + " " + '{:5.2f}'.format(conf))
    return label, rect


if __name__ == "__main__":
    i = 0
    duree = []
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    profile_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')
    reco = cv2.face.LBPHFaceRecognizer_create()
    reco.read("trainner.yml")
    with open("labels.pickle", "rb") as f:
        og_labels = pickle.load(f)
        labels = {v: k for k, v in og_labels.items()}
    cap = cv2.VideoCapture(0)
    width = int(cap.get(3))
    while True:
        print(i)
        d = time.time()
        tickmark = cv2.getTickCount()
        if i % 5 == 0:
            label, rect = reco_vd()
            f = time.time()
            duree.append(f - d)
            print(label, rect)
        i += 1
        if i == 1000:
            break

# Variable global rect et label pour pouvoir les envoyer au flux plus simplement dans le flux
